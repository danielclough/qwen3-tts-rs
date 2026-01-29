//! ECAPA-TDNN Speaker Encoder.
//!
//! Extracts speaker embeddings from mel spectrograms for voice cloning.
//!
//! Architecture:
//! 1. Initial TDNN layer
//! 2. SE-Res2Net blocks (3 layers)
//! 3. Multi-layer Feature Aggregation (MFA)
//! 4. Attentive Statistics Pooling (ASP)
//! 5. Final linear projection

use candle_core::{D, DType, IndexOp, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder, conv1d};

use crate::config::speaker_encoder_config::SpeakerEncoderConfig;

/// Time Delay Network block with Conv1d + ReLU.
/// Uses reflect padding to match PyTorch's padding_mode="reflect".
#[derive(Debug, Clone)]
pub struct TimeDelayNetBlock {
    conv: Conv1d,
    /// Padding amount for "same" convolution
    padding: usize,
}

impl TimeDelayNetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Calculate padding for "same" output length
        // For kernel_size=k, dilation=d: padding = (k-1)*d/2 on each side
        let padding = (kernel_size - 1) * dilation / 2;

        // Use padding=0 in conv, we'll apply reflect padding manually
        let config = Conv1dConfig {
            padding: 0,
            dilation,
            ..Default::default()
        };
        let conv = conv1d(
            in_channels,
            out_channels,
            kernel_size,
            config,
            vb.pp("conv"),
        )?;
        Ok(Self { conv, padding })
    }

    pub fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(in_channels, out_channels, kernel_size, dilation, vb)
    }
}

/// Apply 1D reflect padding to a tensor of shape (batch, channels, length).
pub fn reflect_pad_1d(xs: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    if pad_left == 0 && pad_right == 0 {
        return Ok(xs.clone());
    }

    // Ensure input is contiguous
    let xs = xs.contiguous()?;
    let (_batch, _channels, length) = xs.dims3()?;
    let device = xs.device();

    // Build indices for the padded output
    // For reflect padding: [pad_left, pad_left-1, ..., 1, 0, 1, 2, ..., length-1, length-2, length-3, ...]
    let mut indices = Vec::with_capacity(pad_left + length + pad_right);

    // Left padding: indices pad_left, pad_left-1, ..., 1 (reversed)
    for i in (1..=pad_left).rev() {
        indices.push(i as u32);
    }

    // Original sequence: 0, 1, 2, ..., length-1
    for i in 0..length {
        indices.push(i as u32);
    }

    // Right padding: indices length-2, length-3, ... (reversed from end)
    for i in 0..pad_right {
        indices.push((length - 2 - i) as u32);
    }

    let indices_tensor = Tensor::from_vec(indices, pad_left + length + pad_right, device)?;

    // index_select on dimension 2
    xs.index_select(&indices_tensor, 2)
}

impl Module for TimeDelayNetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Apply reflect padding
        let padded = reflect_pad_1d(xs, self.padding, self.padding)?;
        // Apply conv (with padding=0) and ReLU
        self.conv.forward(&padded)?.relu()
    }
}

/// Res2Net block for multi-scale feature extraction.
#[derive(Debug, Clone)]
pub struct Res2NetBlock {
    blocks: Vec<TimeDelayNetBlock>,
    scale: usize,
}

impl Res2NetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_channel = in_channels / scale;
        let hidden_channel = out_channels / scale;

        let blocks = (0..(scale - 1))
            .map(|i| {
                TimeDelayNetBlock::new(
                    in_channel,
                    hidden_channel,
                    kernel_size,
                    dilation,
                    vb.pp(format!("blocks.{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { blocks, scale })
    }

    pub fn load(
        in_channels: usize,
        out_channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(in_channels, out_channels, scale, kernel_size, dilation, vb)
    }
}

impl Module for Res2NetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Split input into scale parts along channel dimension
        let channels = xs.dim(1)?;
        let chunk_size = channels / self.scale;

        let mut outputs = Vec::with_capacity(self.scale);
        let mut prev_output: Option<Tensor> = None;

        for i in 0..self.scale {
            let hidden_part = xs.narrow(1, i * chunk_size, chunk_size)?;

            let output_part = if i == 0 {
                hidden_part
            } else if i == 1 {
                self.blocks[i - 1].forward(&hidden_part)?
            } else {
                let combined = (hidden_part + prev_output.as_ref().unwrap())?;
                self.blocks[i - 1].forward(&combined)?
            };

            prev_output = Some(output_part.clone());
            outputs.push(output_part);
        }

        Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)
    }
}

/// Squeeze-and-Excitation block for channel attention.
#[derive(Debug, Clone)]
pub struct SqueezeExcitationBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SqueezeExcitationBlock {
    pub fn new(
        in_channels: usize,
        se_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: 0,
            ..Default::default()
        };
        let conv1 = conv1d(in_channels, se_channels, 1, config, vb.pp("conv1"))?;
        let conv2 = conv1d(se_channels, out_channels, 1, config, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    pub fn load(
        in_channels: usize,
        se_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(in_channels, se_channels, out_channels, vb)
    }
}

impl Module for SqueezeExcitationBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Global average pooling
        let xs_mean = xs.mean_keepdim(2)?;

        // Squeeze and excitation
        let scale = self.conv1.forward(&xs_mean)?.relu()?;
        let scale = candle_nn::ops::sigmoid(&self.conv2.forward(&scale)?)?;

        // Apply scale
        xs.broadcast_mul(&scale)
    }
}

/// Combined SE-Res2Net block used in ECAPA-TDNN.
#[derive(Debug, Clone)]
pub struct SqueezeExcitationRes2NetBlock {
    tdnn1: TimeDelayNetBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TimeDelayNetBlock,
    se_block: SqueezeExcitationBlock,
    out_channels: usize,
}

impl SqueezeExcitationRes2NetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        res2net_scale: usize,
        se_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let tdnn1 = TimeDelayNetBlock::new(in_channels, out_channels, 1, 1, vb.pp("tdnn1"))?;
        let res2net_block = Res2NetBlock::new(
            out_channels,
            out_channels,
            res2net_scale,
            kernel_size,
            dilation,
            vb.pp("res2net_block"),
        )?;
        let tdnn2 = TimeDelayNetBlock::new(out_channels, out_channels, 1, 1, vb.pp("tdnn2"))?;
        let se_block = SqueezeExcitationBlock::new(
            out_channels,
            se_channels,
            out_channels,
            vb.pp("se_block"),
        )?;

        Ok(Self {
            tdnn1,
            res2net_block,
            tdnn2,
            se_block,
            out_channels,
        })
    }

    pub fn load(
        in_channels: usize,
        out_channels: usize,
        res2net_scale: usize,
        se_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(
            in_channels,
            out_channels,
            res2net_scale,
            se_channels,
            kernel_size,
            dilation,
            vb,
        )
    }
}

impl Module for SqueezeExcitationRes2NetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;

        let hidden = self.tdnn1.forward(xs)?;
        let hidden = self.res2net_block.forward(&hidden)?;
        let hidden = self.tdnn2.forward(&hidden)?;
        let hidden = self.se_block.forward(&hidden)?;

        // Residual connection (may need projection if channels differ)
        if residual.dim(1)? == self.out_channels {
            residual + hidden
        } else {
            Ok(hidden)
        }
    }
}

/// Attentive Statistics Pooling for speaker embedding.
///
/// Computes weighted mean and standard deviation of features.
/// Supports variable-length sequences through masking.
#[derive(Debug, Clone)]
pub struct AttentiveStatisticsPooling {
    tdnn: TimeDelayNetBlock,
    conv: Conv1d,
    eps: f64,
}

/// Creates a binary mask for variable-length sequences.
///
/// # Arguments
/// * `lengths` - Tensor of shape (batch,) containing the length of each sequence
/// * `max_len` - Maximum sequence length
/// * `device` - Device to create the mask on
/// * `dtype` - Data type for the mask
///
/// # Returns
/// A mask of shape (batch, max_len) where positions < length are 1.0, else 0.0
fn length_to_mask(lengths: &Tensor, max_len: usize, dtype: DType) -> Result<Tensor> {
    let device = lengths.device();
    let batch_size = lengths.dim(0)?;

    // Create range [0, 1, 2, ..., max_len-1]
    let range = Tensor::arange(0u32, max_len as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)? // (1, max_len)
        .broadcast_as((batch_size, max_len))?;

    // Convert lengths to f32 and broadcast
    let lengths_f32 = lengths.to_dtype(DType::F32)?.unsqueeze(1)?; // (batch, 1)

    // mask[i, j] = 1.0 if j < lengths[i] else 0.0
    let mask = range.lt(&lengths_f32)?;
    mask.to_dtype(dtype)
}

impl AttentiveStatisticsPooling {
    pub fn new(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        // TDNN takes concatenated [hidden, mean, std]
        let tdnn = TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?;
        let config = Conv1dConfig {
            padding: 0,
            ..Default::default()
        };
        let conv = conv1d(attention_channels, channels, 1, config, vb.pp("conv"))?;
        Ok(Self {
            tdnn,
            conv,
            eps: 1e-12,
        })
    }

    pub fn load(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(channels, attention_channels, vb)
    }

    fn compute_statistics(&self, xs: &Tensor, weights: &Tensor) -> Result<(Tensor, Tensor)> {
        // Weighted mean: sum(weights * x, dim=2)
        let mean = (xs.broadcast_mul(weights)?).sum(2)?; // (batch, channels)

        // Weighted std: sqrt(sum(weights * (x - mean)^2, dim=2))
        let mean_expanded = mean.unsqueeze(2)?; // (batch, channels, 1)
        let diff = xs.broadcast_sub(&mean_expanded)?;
        let variance = diff.sqr()?.broadcast_mul(weights)?.sum(2)?;
        let std = (variance + self.eps)?.sqrt()?;

        Ok((mean, std))
    }

    /// Forward pass with explicit sequence lengths for proper masking.
    ///
    /// # Arguments
    /// * `xs` - Input tensor of shape (batch, channels, seq_len)
    /// * `lengths` - Optional tensor of shape (batch,) with actual sequence lengths.
    ///   If None, assumes all sequences are full length.
    ///
    /// # Returns
    /// Pooled statistics of shape (batch, channels*2, 1)
    pub fn forward_with_lengths(&self, xs: &Tensor, lengths: Option<&Tensor>) -> Result<Tensor> {
        let seq_length = xs.dim(2)?;
        let batch_size = xs.dim(0)?;
        let dtype = xs.dtype();
        let device = xs.device();

        // Create mask based on lengths
        let mask = if let Some(lens) = lengths {
            length_to_mask(lens, seq_length, dtype)?
        } else {
            // All ones if no lengths provided (full sequences)
            Tensor::ones((batch_size, seq_length), dtype, device)?
        };

        // Expand mask to (batch, 1, seq_len) for broadcasting with (batch, channels, seq_len)
        let mask = mask.unsqueeze(1)?;

        // Normalize mask so it sums to 1 for weighted statistics
        let total = mask.sum_keepdim(2)?;
        let normalized_mask = mask.broadcast_div(&total)?;

        // Compute initial mean and std using normalized mask
        let (mean, std) = self.compute_statistics(xs, &normalized_mask)?;

        // Expand mean and std to full sequence length for attention input
        let mean_expanded = mean.unsqueeze(2)?.repeat((1, 1, seq_length))?;
        let std_expanded = std.unsqueeze(2)?.repeat((1, 1, seq_length))?;

        // Concatenate [hidden, mean, std] along channel dimension
        let attention_input = Tensor::cat(&[xs, &mean_expanded, &std_expanded], 1)?;

        // Compute attention weights through TDNN -> tanh -> conv
        let attention = self.tdnn.forward(&attention_input)?;
        let attention = attention.tanh()?;
        let attention = self.conv.forward(&attention)?;

        // CRITICAL: Mask out padded positions before softmax
        // Fill masked positions (mask == 0) with -inf so they get ~0 attention after softmax
        // where_cond requires U32 condition and matching shapes (no broadcast)
        let attention_channels = attention.dim(1)?;
        // Convert mask to U32 for where_cond and expand to match attention shape
        let mask_u32 = mask.to_dtype(candle_core::DType::U32)?;
        let mask_expanded = mask_u32.expand((batch_size, attention_channels, seq_length))?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, attention.shape(), device)?;
        let attention_f32 = attention.to_dtype(candle_core::DType::F32)?;
        let attention = mask_expanded.where_cond(&attention_f32, &neg_inf)?.to_dtype(dtype)?;

        // Softmax over time dimension
        let attention = candle_nn::ops::softmax_last_dim(&attention)?;

        // Compute final statistics using attention weights
        let (mean_final, std_final) = self.compute_statistics(xs, &attention)?;

        // Concatenate mean and std: (batch, channels*2)
        let pooled_stats = Tensor::cat(&[&mean_final, &std_final], 1)?;

        // Add time dimension: (batch, channels*2, 1)
        pooled_stats.unsqueeze(2)
    }
}

impl Module for AttentiveStatisticsPooling {
    /// Forward pass assuming full-length sequences (no padding).
    ///
    /// For variable-length sequences, use `forward_with_lengths()` instead.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Delegate to forward_with_lengths with no explicit lengths (assumes full sequences)
        self.forward_with_lengths(xs, None)
    }
}

/// ECAPA-TDNN Speaker Encoder.
///
/// Extracts a fixed-size speaker embedding from variable-length mel spectrograms.
#[derive(Debug, Clone)]
pub struct SpeakerEncoder {
    initial_tdnn: TimeDelayNetBlock,
    se_res2net_blocks: Vec<SqueezeExcitationRes2NetBlock>,
    mfa: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
}

impl SpeakerEncoder {
    pub fn new(config: &SpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Initial TDNN layer
        let initial_tdnn = TimeDelayNetBlock::new(
            config.mel_dim,
            config.enc_channels[0],
            config.enc_kernel_sizes[0],
            config.enc_dilations[0],
            vb.pp("blocks.0"),
        )?;

        // SE-Res2Net blocks
        let mut se_res2net_blocks = Vec::new();
        for i in 1..(config.enc_channels.len() - 1) {
            let block = SqueezeExcitationRes2NetBlock::new(
                config.enc_channels[i - 1],
                config.enc_channels[i],
                config.enc_res2net_scale,
                config.enc_se_channels,
                config.enc_kernel_sizes[i],
                config.enc_dilations[i],
                vb.pp(format!("blocks.{}", i)),
            )?;
            se_res2net_blocks.push(block);
        }

        // Multi-layer Feature Aggregation
        // MFA takes concatenated outputs from all SE-Res2Net blocks
        let mfa_in_channels: usize = config.enc_channels[..(config.enc_channels.len() - 1)]
            .iter()
            .skip(1)
            .sum();
        let mfa = TimeDelayNetBlock::new(
            mfa_in_channels,
            config.enc_channels[config.enc_channels.len() - 1],
            config.enc_kernel_sizes[config.enc_kernel_sizes.len() - 1],
            config.enc_dilations[config.enc_dilations.len() - 1],
            vb.pp("mfa"),
        )?;

        // Attentive Statistics Pooling
        let asp = AttentiveStatisticsPooling::new(
            config.enc_channels[config.enc_channels.len() - 1],
            config.enc_attention_channels,
            vb.pp("asp"),
        )?;

        // Final projection
        let fc_config = Conv1dConfig {
            padding: 0,
            ..Default::default()
        };
        let fc = conv1d(
            config.enc_channels[config.enc_channels.len() - 1] * 2, // *2 because of mean+std
            config.enc_dim,
            1,
            fc_config,
            vb.pp("fc"),
        )?;

        Ok(Self {
            initial_tdnn,
            se_res2net_blocks,
            mfa,
            asp,
            fc,
        })
    }

    pub fn load(config: &SpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::new(config, vb)
    }
}

impl SpeakerEncoder {
    /// Forward pass with explicit sequence lengths for proper masking.
    ///
    /// # Arguments
    /// * `xs` - Input mel spectrogram of shape (batch, time, mel_dim)
    /// * `lengths` - Optional tensor of shape (batch,) with actual sequence lengths.
    ///   If None, assumes all sequences are full length.
    ///
    /// # Returns
    /// Speaker embedding of shape (batch, enc_dim)
    pub fn forward_with_lengths(&self, xs: &Tensor, lengths: Option<&Tensor>) -> Result<Tensor> {
        // Transpose to (batch, mel_dim, time) for conv1d and make contiguous
        let hidden = xs.transpose(1, 2)?.contiguous()?;
        tracing::debug!(shape = ?hidden.shape(), "speaker_encoder input after transpose");

        // Initial TDNN
        let mut hidden = self.initial_tdnn.forward(&hidden)?;
        let mut hidden_states = vec![hidden.clone()];
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(h) = hidden.to_dtype(DType::F32)
            && let (Ok(min), Ok(max), Ok(mean), Ok(first5)) = (
                h.min(D::Minus1)
                    .and_then(|t| t.min(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.max(D::Minus1)
                    .and_then(|t| t.max(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.mean_all().and_then(|t| t.to_scalar::<f32>()),
                h.i((0, ..5, 0)).and_then(|t| t.to_vec1::<f32>()),
            )
        {
            tracing::debug!(
                shape = ?hidden.shape(),
                min = format!("{:.4}", min),
                max = format!("{:.4}", max),
                mean_val = format!("{:.6}", mean),
                first5 = ?first5,
                "After blocks[0]"
            );
        }

        // SE-Res2Net blocks
        for (i, block) in self.se_res2net_blocks.iter().enumerate() {
            hidden = block.forward(&hidden)?;
            hidden_states.push(hidden.clone());
            if tracing::enabled!(tracing::Level::DEBUG)
                && let Ok(h) = hidden.to_dtype(DType::F32)
                && let (Ok(min), Ok(max), Ok(mean), Ok(first5)) = (
                    h.min(D::Minus1)
                        .and_then(|t| t.min(D::Minus1))
                        .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                    h.max(D::Minus1)
                        .and_then(|t| t.max(D::Minus1))
                        .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                    h.mean_all().and_then(|t| t.to_scalar::<f32>()),
                    h.i((0, ..5, 0)).and_then(|t| t.to_vec1::<f32>()),
                )
            {
                tracing::debug!(
                    block_idx = i + 1,
                    shape = ?hidden.shape(),
                    min = format!("{:.4}", min),
                    max = format!("{:.4}", max),
                    mean_val = format!("{:.6}", mean),
                    first5 = ?first5,
                    "After SE-Res2Net block"
                );
            }
        }

        // Multi-layer Feature Aggregation
        let mfa_input = Tensor::cat(&hidden_states[1..].iter().collect::<Vec<_>>(), 1)?;
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(h) = mfa_input.to_dtype(DType::F32)
            && let (Ok(min), Ok(max), Ok(mean)) = (
                h.min(D::Minus1)
                    .and_then(|t| t.min(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.max(D::Minus1)
                    .and_then(|t| t.max(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.mean_all().and_then(|t| t.to_scalar::<f32>()),
            )
        {
            tracing::debug!(
                shape = ?mfa_input.shape(),
                min = format!("{:.4}", min),
                max = format!("{:.4}", max),
                mean_val = format!("{:.6}", mean),
                "After cat(hidden_states[1:])"
            );
        }

        let hidden = self.mfa.forward(&mfa_input)?;
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(h) = hidden.to_dtype(DType::F32)
            && let (Ok(min), Ok(max), Ok(mean), Ok(first5)) = (
                h.min(D::Minus1)
                    .and_then(|t| t.min(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.max(D::Minus1)
                    .and_then(|t| t.max(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.mean_all().and_then(|t| t.to_scalar::<f32>()),
                h.i((0, ..5, 0)).and_then(|t| t.to_vec1::<f32>()),
            )
        {
            tracing::debug!(
                shape = ?hidden.shape(),
                min = format!("{:.4}", min),
                max = format!("{:.4}", max),
                mean_val = format!("{:.6}", mean),
                first5 = ?first5,
                "After MFA"
            );
        }

        // Attentive Statistics Pooling with masking support
        let hidden = self.asp.forward_with_lengths(&hidden, lengths)?;
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(h) = hidden.to_dtype(DType::F32)
            && let (Ok(min), Ok(max), Ok(mean), Ok(first10)) = (
                h.min(D::Minus1)
                    .and_then(|t| t.min(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.max(D::Minus1)
                    .and_then(|t| t.max(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.mean_all().and_then(|t| t.to_scalar::<f32>()),
                h.i((0, ..10, 0)).and_then(|t| t.to_vec1::<f32>()),
            )
        {
            tracing::debug!(
                shape = ?hidden.shape(),
                min = format!("{:.4}", min),
                max = format!("{:.4}", max),
                mean_val = format!("{:.6}", mean),
                first10 = ?first10,
                "After ASP"
            );
        }

        // Final projection
        let hidden = self.fc.forward(&hidden)?;
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(h) = hidden.to_dtype(DType::F32)
            && let (Ok(min), Ok(max), Ok(mean)) = (
                h.min(D::Minus1)
                    .and_then(|t| t.min(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.max(D::Minus1)
                    .and_then(|t| t.max(D::Minus1))
                    .and_then(|t| Ok(t.to_vec1::<f32>()?[0])),
                h.mean_all().and_then(|t| t.to_scalar::<f32>()),
            )
        {
            tracing::debug!(
                shape = ?hidden.shape(),
                min = format!("{:.4}", min),
                max = format!("{:.4}", max),
                mean_val = format!("{:.6}", mean),
                "After FC"
            );
        }

        // Squeeze the time dimension (should be 1 after pooling)
        hidden.squeeze(2)
    }
}

impl Module for SpeakerEncoder {
    /// Forward pass assuming full-length sequences.
    ///
    /// Input: (batch, time, mel_dim) - mel spectrogram
    /// Output: (batch, enc_dim) - speaker embedding
    ///
    /// For variable-length sequences, use `forward_with_lengths()` instead.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_lengths(xs, None)
    }
}
