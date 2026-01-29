//! Whisper-style encoder for the V1 (25Hz) tokenizer.
//!
//! Ref: `Qwen3-TTS/.../vq/whisper_encoder.py`

use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sinusoidal positional embeddings (sin/cos interleaved by half).
///
/// Returns shape `(length, channels)` with `sin` in `[:, :channels/2]` and
/// `cos` in `[:, channels/2:]`.
pub fn sinusoids(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    assert!(channels.is_multiple_of(2), "channels must be even");
    let half = channels / 2;
    let log_timescale_increment = (10000f64).ln() / (half as f64 - 1.0);

    let inv: Vec<f32> = (0..half)
        .map(|i| (-log_timescale_increment * i as f64).exp() as f32)
        .collect();
    let inv = Tensor::from_vec(inv, (1, half), device)?;

    let positions: Vec<f32> = (0..length).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, (length, 1), device)?;

    let scaled = positions.matmul(&inv)?; // (length, half)
    let sin_part = scaled.sin()?;
    let cos_part = scaled.cos()?;
    Tensor::cat(&[&sin_part, &cos_part], D::Minus1)
}

/// Output length after the two CNN layers (conv1 k=3 p=1 s=1, conv2 k=3 p=1 s=2).
pub fn get_t_after_cnn(l_in: usize) -> usize {
    // conv1: padding=1, kernel=3, stride=1 → l_out = l_in
    let l = l_in;
    // conv2: padding=1, kernel=3, stride=2
    (l + 2 - (3 - 1) - 1) / 2 + 1
}

/// Simple 1-D average pooling implemented via reshape.
///
/// `x`: `(T, D)`, returns `(T / stride, D)` — truncates if T is not divisible.
pub fn avg_pool1d(x: &Tensor, kernel: usize, stride: usize) -> Result<Tensor> {
    assert_eq!(kernel, stride, "only kernel == stride supported");
    let (t, d) = x.dims2()?;
    let t_out = t / stride;
    // Truncate, reshape to (t_out, stride, d), mean over dim 1
    let x = x.narrow(0, 0, t_out * stride)?;
    let x = x.reshape((t_out, stride, d))?;
    x.mean(1)
}

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

/// Multi-head attention with varlen (cu_seqlens) support.
///
/// Q has bias, K has no bias, V has bias, Out has bias.
#[derive(Debug, Clone)]
pub struct WhisperV1MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl WhisperV1MultiHeadAttention {
    pub fn new(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let query = candle_nn::linear(n_state, n_state, vb.pp("query"))?;
        let key = candle_nn::linear_no_bias(n_state, n_state, vb.pp("key"))?;
        let value = candle_nn::linear(n_state, n_state, vb.pp("value"))?;
        let out = candle_nn::linear(n_state, n_state, vb.pp("out"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
        })
    }

    /// Forward with packed sequences.
    ///
    /// `x`: `(total_tokens, n_state)` packed tensor
    /// `cu_seqlens`: `(num_seqs + 1,)` cumulative sequence lengths
    pub fn forward(&self, x: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        let attn = self.qkv_attention_manual(&q, &k, &v, cu_seqlens)?;
        self.out.forward(&attn)
    }

    fn qkv_attention_manual(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cu_seqlens: &[usize],
    ) -> Result<Tensor> {
        let (n_ctx, n_state) = q.dims2()?;
        let head_dim = n_state / self.n_head;
        let scale = (head_dim as f64).powf(-0.5);

        let q = q.reshape((n_ctx, self.n_head, head_dim))?;
        let k = k.reshape((n_ctx, self.n_head, head_dim))?;
        let v = v.reshape((n_ctx, self.n_head, head_dim))?;

        let batch_size = cu_seqlens.len() - 1;
        let seqlens: Vec<usize> = (0..batch_size)
            .map(|i| cu_seqlens[i + 1] - cu_seqlens[i])
            .collect();
        let max_seqlen = *seqlens.iter().max().unwrap_or(&1);

        let dev = q.device();
        let dtype = q.dtype();

        // Pad into (batch, max_seqlen, n_head, head_dim)
        let mut q_parts = Vec::with_capacity(batch_size);
        let mut k_parts = Vec::with_capacity(batch_size);
        let mut v_parts = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start = cu_seqlens[i];
            let seq_len = seqlens[i];
            let q_i = q.narrow(0, start, seq_len)?;
            let k_i = k.narrow(0, start, seq_len)?;
            let v_i = v.narrow(0, start, seq_len)?;

            if seq_len < max_seqlen {
                let pad_len = max_seqlen - seq_len;
                let pad = Tensor::zeros((pad_len, self.n_head, head_dim), dtype, dev)?;
                q_parts.push(Tensor::cat(&[&q_i, &pad], 0)?);
                k_parts.push(Tensor::cat(&[&k_i, &pad], 0)?);
                v_parts.push(Tensor::cat(&[&v_i, &pad], 0)?);
            } else {
                q_parts.push(q_i);
                k_parts.push(k_i);
                v_parts.push(v_i);
            }
        }

        // Stack → (batch, max_seqlen, n_head, head_dim)
        let q_padded = Tensor::stack(&q_parts, 0)?;
        let k_padded = Tensor::stack(&k_parts, 0)?;
        let v_padded = Tensor::stack(&v_parts, 0)?;

        // Transpose to (batch, n_head, max_seqlen, head_dim)
        let q_padded = q_padded.transpose(1, 2)?;
        let k_padded = k_padded.transpose(1, 2)?;
        let v_padded = v_padded.transpose(1, 2)?;

        // Attention mask: (batch, 1, 1, max_seqlen) — True where valid
        let mut mask_data = Vec::with_capacity(batch_size * max_seqlen);
        for &sl in &seqlens {
            for j in 0..max_seqlen {
                mask_data.push(if j < sl { 0.0f32 } else { f32::NEG_INFINITY });
            }
        }
        let attn_mask =
            Tensor::from_vec(mask_data, (batch_size, 1, 1, max_seqlen), dev)?
                .to_dtype(dtype)?;

        // Scaled dot-product: (batch, n_head, max_seqlen, max_seqlen)
        let attn_scores = q_padded.contiguous()?.matmul(&k_padded.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let attn_scores = (attn_scores * scale)?;
        let attn_scores = attn_scores.broadcast_add(&attn_mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;

        // Context: (batch, n_head, max_seqlen, head_dim) → (batch, max_seqlen, n_state)
        let context = attn_weights.matmul(&v_padded.contiguous()?)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        let context = context.reshape((batch_size, max_seqlen, n_state))?;

        // Unpad
        let mut output_parts = Vec::with_capacity(batch_size);
        for (i, &sl) in seqlens.iter().enumerate() {
            output_parts.push(context.i(i)?.narrow(0, 0, sl)?);
        }
        Tensor::cat(&output_parts, 0)
    }
}

// ---------------------------------------------------------------------------
// ResidualAttentionBlock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WhisperV1ResidualAttentionBlock {
    attn_ln: LayerNorm,
    attn: WhisperV1MultiHeadAttention,
    mlp_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl WhisperV1ResidualAttentionBlock {
    pub fn new(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let attn_ln = candle_nn::layer_norm(n_state, 1e-5, vb.pp("attn_ln"))?;
        let attn = WhisperV1MultiHeadAttention::new(n_state, n_head, vb.pp("attn"))?;
        let mlp_ln = candle_nn::layer_norm(n_state, 1e-5, vb.pp("mlp_ln"))?;
        let n_mlp = n_state * 4;
        let fc1 = candle_nn::linear(n_state, n_mlp, vb.pp("mlp.0"))?;
        let fc2 = candle_nn::linear(n_mlp, n_state, vb.pp("mlp.2"))?;
        Ok(Self {
            attn_ln,
            attn,
            mlp_ln,
            fc1,
            fc2,
        })
    }

    pub fn forward(&self, x: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let residual = x;
        let x = (residual + self.attn.forward(&self.attn_ln.forward(x)?, cu_seqlens)?)?;
        let residual = &x;
        let mlp_out = self.fc1.forward(&self.mlp_ln.forward(&x)?)?;
        let mlp_out = mlp_out.gelu_erf()?;
        let mlp_out = self.fc2.forward(&mlp_out)?;
        residual + mlp_out
    }
}

// ---------------------------------------------------------------------------
// WhisperV1Encoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WhisperV1Encoder {
    conv1: Conv1d,
    conv2: Conv1d,
    pub positional_embedding: Tensor,
    pub blocks: Vec<WhisperV1ResidualAttentionBlock>,
    ln_post: LayerNorm,
    proj: Linear,
    audio_bos_eos_token: Embedding,
    pub n_window: usize,
    pub n_state: usize,
    pub output_dim: usize,
}

impl WhisperV1Encoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_mels: usize,
        n_ctx: usize,
        n_state: usize,
        n_head: usize,
        n_layer: usize,
        n_window: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = candle_nn::conv1d(
            n_mels,
            n_state,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            n_state,
            n_state,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        let positional_embedding = vb.get((n_ctx, n_state), "positional_embedding")?;

        let blocks = (0..n_layer)
            .map(|i| {
                WhisperV1ResidualAttentionBlock::new(n_state, n_head, vb.pp(format!("blocks.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let ln_post = candle_nn::layer_norm(n_state, 1e-5, vb.pp("ln_post"))?;
        let proj = candle_nn::linear_no_bias(n_state, output_dim, vb.pp("proj"))?;
        let audio_bos_eos_token = candle_nn::embedding(2, output_dim, vb.pp("audio_bos_eos_token"))?;

        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            proj,
            audio_bos_eos_token,
            n_window,
            n_state,
            output_dim,
        })
    }

    /// Run the CNN front-end on a single mel chunk: conv1→GELU→conv2→GELU→permute→add PE.
    ///
    /// `chunk`: `(n_mels, chunk_frames)` — a single mel spectrogram chunk.
    /// Returns `(chunk_after_cnn_len, n_state)`.
    pub fn cnn_and_pe(&self, chunk: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(&chunk.unsqueeze(0)?)?.squeeze(0)?;
        let x = x.gelu_erf()?;
        let x = self.conv2.forward(&x.unsqueeze(0)?)?.squeeze(0)?;
        let x = x.gelu_erf()?;
        // (n_state, T) → (T, n_state)
        let x = x.t()?;
        let chunk_len = x.dim(0)?;
        let pe = self.positional_embedding.narrow(0, 0, chunk_len)?;
        let pe = pe.to_dtype(x.dtype())?;
        x + pe
    }

    /// Full forward pass matching the Python `WhisperEncoder.forward`.
    ///
    /// `x_list`: list of mel spectrograms, each `(n_mels, T_i)`.
    /// `audio_aftercnnlens`: after-CNN lengths per audio.
    /// `audio_seqlens`: output sequence lengths including BOS/EOS per audio.
    pub fn forward(
        &self,
        x_list: &[Tensor],
        audio_aftercnnlens: &[usize],
        audio_seqlens: &[usize],
    ) -> Result<Tensor> {
        // Step 1-2: Per mel, split into chunks of n_window*2 frames, CNN + PE
        let mut aftercnn_x_list = Vec::new();
        for each_x in x_list {
            let total_frames = each_x.dim(1)?;
            let chunk_size = self.n_window * 2;
            let mut offset = 0;
            while offset < total_frames {
                let end = (offset + chunk_size).min(total_frames);
                let chunk = each_x.narrow(1, offset, end - offset)?;
                aftercnn_x_list.push(self.cnn_and_pe(&chunk)?);
                offset = end;
            }
        }

        // Cat all chunks: (total_tokens, n_state)
        let mut x = Tensor::cat(&aftercnn_x_list, 0)?;

        // Step 4: Compute cu_seqlens from audio_aftercnnlens
        let cu_seqlens = Self::compute_cu_seqlens(audio_aftercnnlens, self.n_window);

        // Step 5: Run all blocks
        for block in &self.blocks {
            x = block.forward(&x, &cu_seqlens)?;
        }

        // Step 6: Split by audio_aftercnnlens, avg_pool1d(stride=2) each, re-cat
        let mut pooled_parts = Vec::new();
        let mut pos = 0;
        for &acl in audio_aftercnnlens {
            let part = x.narrow(0, pos, acl)?;
            // Permute to (n_state, T), avg_pool, permute back
            let pooled = avg_pool1d(&part, 2, 2)?;
            pooled_parts.push(pooled);
            pos += acl;
        }
        let x = Tensor::cat(&pooled_parts, 0)?;

        // Step 7: ln_post → proj
        let x = self.ln_post.forward(&x)?;
        let x = self.proj.forward(&x)?;

        // Step 8: Insert BOS/EOS tokens
        self.insert_bos_eos(&x, audio_seqlens)
    }

    /// Compute cu_seqlens: split each audio_aftercnnlen into ≤ n_window segments.
    pub fn compute_cu_seqlens(audio_aftercnnlens: &[usize], n_window: usize) -> Vec<usize> {
        let mut output_list = Vec::new();
        for &item in audio_aftercnnlens {
            let mut remaining = item;
            while remaining > n_window {
                output_list.push(n_window);
                remaining -= n_window;
            }
            output_list.push(remaining);
        }
        let mut cu_seqlens = Vec::with_capacity(output_list.len() + 1);
        cu_seqlens.push(0);
        let mut acc = 0;
        for len in output_list {
            acc += len;
            cu_seqlens.push(acc);
        }
        cu_seqlens
    }

    /// Insert BOS/EOS tokens at the correct positions in the output.
    fn insert_bos_eos(&self, x: &Tensor, audio_seqlens: &[usize]) -> Result<Tensor> {
        let dtype = x.dtype();
        let dev = x.device();

        let bos = self.audio_bos_eos_token.forward(&Tensor::new(&[0u32], dev)?)?
            .squeeze(0)?
            .to_dtype(dtype)?;
        let eos = self.audio_bos_eos_token.forward(&Tensor::new(&[1u32], dev)?)?
            .squeeze(0)?
            .to_dtype(dtype)?;

        // Build output by interleaving BOS, content, EOS per audio
        let mut parts = Vec::new();
        let mut x_offset = 0;

        for &seqlen in audio_seqlens {
            let content_len = seqlen - 2;
            parts.push(bos.unsqueeze(0)?);
            if content_len > 0 {
                parts.push(x.narrow(0, x_offset, content_len)?);
                x_offset += content_len;
            }
            parts.push(eos.unsqueeze(0)?);
        }

        Tensor::cat(&parts, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_sinusoids_shape_and_values() -> Result<()> {
        let s = sinusoids(8, 4, &Device::Cpu)?;
        assert_eq!(s.dims(), &[8, 4]);

        let row0 = s.i(0)?.to_vec1::<f32>()?;
        // sin(0) = 0 for both sin columns
        assert!((row0[0]).abs() < 1e-6, "sin(0) should be 0, got {}", row0[0]);
        assert!((row0[1]).abs() < 1e-6, "sin(0) should be 0, got {}", row0[1]);
        // cos(0) = 1 for both cos columns
        assert!((row0[2] - 1.0).abs() < 1e-6, "cos(0) should be 1, got {}", row0[2]);
        assert!((row0[3] - 1.0).abs() < 1e-6, "cos(0) should be 1, got {}", row0[3]);
        Ok(())
    }

    #[test]
    fn test_get_t_after_cnn() {
        assert_eq!(get_t_after_cnn(100), 50);
        assert_eq!(get_t_after_cnn(200), 100);
        assert_eq!(get_t_after_cnn(3000), 1500);
        assert_eq!(get_t_after_cnn(201), 101);
    }

    #[test]
    fn test_avg_pool1d_manual() -> Result<()> {
        let data = Tensor::new(
            &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            &Device::Cpu,
        )?;
        let pooled = avg_pool1d(&data, 2, 2)?;
        assert_eq!(pooled.dims(), &[2, 2]);
        let vals = pooled.to_vec2::<f32>()?;
        assert!((vals[0][0] - 2.0).abs() < 1e-6); // (1+3)/2
        assert!((vals[0][1] - 3.0).abs() < 1e-6); // (2+4)/2
        assert!((vals[1][0] - 6.0).abs() < 1e-6); // (5+7)/2
        assert!((vals[1][1] - 7.0).abs() < 1e-6); // (6+8)/2
        Ok(())
    }

    #[test]
    fn test_avg_pool1d_odd_length() -> Result<()> {
        // 5 rows → truncate to 4 → pool to 2
        let data = Tensor::new(
            &[
                [1.0f32, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            &Device::Cpu,
        )?;
        let pooled = avg_pool1d(&data, 2, 2)?;
        assert_eq!(pooled.dims(), &[2, 2]);
        let vals = pooled.to_vec2::<f32>()?;
        assert!((vals[0][0] - 1.5).abs() < 1e-6); // (1+2)/2
        assert!((vals[1][0] - 3.5).abs() < 1e-6); // (3+4)/2
        Ok(())
    }

    #[test]
    fn test_multihead_attention_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let n_state = 16;
        let n_head = 2;

        let vb = VarBuilder::zeros(DType::F32, &dev);
        let attn = WhisperV1MultiHeadAttention::new(n_state, n_head, vb)?;

        let total_tokens = 15;
        let x = Tensor::randn(0f32, 1.0, (total_tokens, n_state), &dev)?;
        let cu_seqlens = vec![0, 5, 10, 15];

        let out = attn.forward(&x, &cu_seqlens)?;
        assert_eq!(out.dims(), &[15, 16]);
        Ok(())
    }
}
