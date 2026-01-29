//! BigVGAN vocoder: mel spectrogram → audio waveform.
//!
//! Uses Snake activation with anti-aliased upsampling/downsampling.

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

use super::bigvgan_activations::TorchActivation1d;
use super::bigvgan_amp_block::AMPBlock;

// ---------------------------------------------------------------------------
// BigVGANDecoder
// ---------------------------------------------------------------------------

/// BigVGAN decoder configuration.
#[derive(Debug, Clone)]
pub struct BigVGANConfig {
    pub mel_dim: usize,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
}

impl Default for BigVGANConfig {
    fn default() -> Self {
        Self {
            mel_dim: 80,
            upsample_initial_channel: 1536,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_rates: vec![5, 3, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![11, 7, 4, 4, 4, 4],
        }
    }
}

/// BigVGAN vocoder: mel spectrogram → audio waveform.
#[derive(Debug)]
pub struct BigVGANDecoder {
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AMPBlock>,
    activation_post: TorchActivation1d,
    conv_post: Conv1d,
    num_upsample_layers: usize,
    num_residual_blocks: usize,
}

impl BigVGANDecoder {
    pub fn new(config: &BigVGANConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let num_residual_blocks = config.resblock_kernel_sizes.len();
        let num_upsample_layers = config.upsample_rates.len();

        let conv_pre_config = Conv1dConfig {
            padding: 2,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv_pre = candle_nn::conv1d(
            config.mel_dim,
            config.upsample_initial_channel,
            5,
            conv_pre_config,
            vb.pp("conv_pre"),
        )?;

        let mut ups = Vec::new();
        for (i, (&stride, &kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = config.upsample_initial_channel / (1 << i);
            let out_ch = config.upsample_initial_channel / (1 << (i + 1));
            let padding = (kernel_size - stride) / 2;
            let config = ConvTranspose1dConfig {
                padding,
                stride,
                dilation: 1,
                output_padding: 0,
                groups: 1,
            };
            let up = candle_nn::conv_transpose1d(
                in_ch,
                out_ch,
                kernel_size,
                config,
                vb.pp(format!("ups.{}.0", i)),
            )?;
            ups.push(up);
        }

        let mut resblocks = Vec::new();
        for layer_idx in 0..num_upsample_layers {
            let ch = config.upsample_initial_channel / (1 << (layer_idx + 1));
            let causal_type = if layer_idx > 1 { "1" } else { "2" };
            for (block_idx, (kernel_size, dilations)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
                .enumerate()
            {
                let flat_idx = layer_idx * num_residual_blocks + block_idx;
                resblocks.push(AMPBlock::new(
                    ch,
                    *kernel_size,
                    dilations,
                    causal_type,
                    vb.pp(format!("resblocks.{}", flat_idx)),
                    &device,
                )?);
            }
        }

        let final_ch = config.upsample_initial_channel / (1 << num_upsample_layers);
        let activation_post =
            TorchActivation1d::new(final_ch, vb.pp("activation_post.act"), &device)?;

        let conv_post_config = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        // conv_post has no bias
        let conv_post_weight = vb.pp("conv_post").get((1, final_ch, 7), "weight")?;
        let conv_post = Conv1d::new(conv_post_weight, None, conv_post_config);

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            num_upsample_layers,
            num_residual_blocks,
        })
    }

    /// Process mel spectrogram: exp → amplitude_to_db → normalize.
    pub fn process_mel_spectrogram(&self, mel: &Tensor) -> Result<Tensor> {
        let amplitude = mel.exp()?;
        let db = amplitude_to_db(&amplitude, -115.0)?;
        let db = (db - 20.0)?;
        normalize_spectrogram(&db, 1.0, -115.0)
    }
}

impl Module for BigVGANDecoder {
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut h = self.process_mel_spectrogram(mel)?;
        h = self.conv_pre.forward(&h)?;

        for layer_idx in 0..self.num_upsample_layers {
            h = self.ups[layer_idx].forward(&h)?;
            let mut res_sum: Option<Tensor> = None;
            for block_idx in 0..self.num_residual_blocks {
                let flat_idx = layer_idx * self.num_residual_blocks + block_idx;
                let block_out = self.resblocks[flat_idx].forward(&h)?;
                res_sum = Some(match res_sum {
                    Some(s) => (&s + &block_out)?,
                    None => block_out,
                });
            }
            h = res_sum
                .unwrap()
                .affine(1.0 / self.num_residual_blocks as f64, 0.0)?;
        }

        h = self.activation_post.forward(&h)?;
        h = self.conv_post.forward(&h)?;
        h.clamp(-1.0, 1.0)?.squeeze(1)
    }
}

fn amplitude_to_db(amplitude: &Tensor, min_db_level: f64) -> Result<Tensor> {
    let min_level = (min_db_level / 20.0 * 10.0f64.ln()).exp();
    let clamped = amplitude.clamp(min_level, f64::MAX)?;
    let log10_factor = 20.0 / 10.0f64.ln();
    clamped.log()?.affine(log10_factor, 0.0)
}

fn normalize_spectrogram(spectrogram: &Tensor, max_value: f64, min_db: f64) -> Result<Tensor> {
    let scale = 2.0 * max_value / (-min_db);
    let offset = -2.0 * max_value * min_db / (-min_db) - max_value;
    spectrogram.affine(scale, offset)?.clamp(-max_value, max_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_total_upsample_factor() {
        let config = BigVGANConfig::default();
        let total: usize = config.upsample_rates.iter().product();
        assert_eq!(total, 240);
    }

    #[test]
    fn test_channel_progression() {
        let config = BigVGANConfig::default();
        let mut ch = config.upsample_initial_channel;
        for _ in 0..config.upsample_rates.len() {
            ch /= 2;
        }
        assert_eq!(ch, 24);
    }

    #[test]
    fn test_process_mel_spectrogram_values() -> Result<()> {
        let device = Device::Cpu;
        // mel=0 → exp(0)=1 → 20*log10(1)=0 → 0-20=-20 → normalize(-20, 1.0, -115.0)
        // scale = 2/115, offset = -2*(-115)/115 - 1 = 2 - 1 = 1
        // result = -20 * 2/115 + 1 = -40/115 + 1 = 75/115 ≈ 0.6522
        let mel = Tensor::zeros((1, 1, 1), DType::F32, &device)?;
        let amplitude = mel.exp()?;
        let db = amplitude_to_db(&amplitude, -115.0)?;
        let db = (db - 20.0)?;
        let result = normalize_spectrogram(&db, 1.0, -115.0)?;
        let val = result.flatten_all()?.to_vec1::<f32>()?[0];
        assert!((val - 0.6522).abs() < 0.01, "mel=0 gave {}, expected ~0.652", val);

        // mel=-5 → exp(-5)≈0.00674 → 20*log10(0.00674)≈-43.42 → -63.42 → normalize
        // result = -63.42 * 2/115 + 1 = -126.84/115 + 1 ≈ -0.103
        let mel = Tensor::new(&[[[-5.0f32]]], &device)?;
        let amplitude = mel.exp()?;
        let db = amplitude_to_db(&amplitude, -115.0)?;
        let db = (db - 20.0)?;
        let result = normalize_spectrogram(&db, 1.0, -115.0)?;
        let val = result.flatten_all()?.to_vec1::<f32>()?[0];
        assert!((val - (-0.103)).abs() < 0.02, "mel=-5 gave {}, expected ~-0.103", val);

        Ok(())
    }

    #[test]
    fn test_process_mel_spectrogram_clamping() -> Result<()> {
        let device = Device::Cpu;
        // Very large mel value should clamp to 1.0
        let mel = Tensor::new(&[[[100.0f32]]], &device)?;
        let amplitude = mel.exp()?;
        let db = amplitude_to_db(&amplitude, -115.0)?;
        let db = (db - 20.0)?;
        let result = normalize_spectrogram(&db, 1.0, -115.0)?;
        let val = result.flatten_all()?.to_vec1::<f32>()?[0];
        assert!((val - 1.0).abs() < 1e-6, "Large mel should clamp to 1.0, got {}", val);

        // Very negative mel should clamp to -1.0
        let mel = Tensor::new(&[[[-1000.0f32]]], &device)?;
        let amplitude = mel.exp()?;
        let db = amplitude_to_db(&amplitude, -115.0)?;
        let db = (db - 20.0)?;
        let result = normalize_spectrogram(&db, 1.0, -115.0)?;
        let val = result.flatten_all()?.to_vec1::<f32>()?[0];
        assert!((val - (-1.0)).abs() < 1e-6, "Very negative mel should clamp to -1.0, got {}", val);

        Ok(())
    }
}
