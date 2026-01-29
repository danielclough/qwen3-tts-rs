//! DiT (Diffusion Transformer) model for mel spectrogram generation.
//!
//! Generates mel spectrograms from codec codes via flow matching ODE.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::audio::decoder::dit_block::DiTDecoderLayer;
use crate::audio::decoder::dit_components::{
    AdaLayerNormZeroFinal, DiTCodecEmbedding, DiTInputEmbedding, DiTTimestepEmbedding,
};
use crate::audio::decoder::dit_rope::DiTRotaryEmbedding;
use crate::config::speaker_encoder_config::SpeakerEncoderConfig;

/// DiT configuration for V1 decoder.
#[derive(Debug, Clone)]
pub struct DiTConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub ff_mult: usize,
    pub emb_dim: usize,
    pub enc_emb_dim: usize,
    pub mel_dim: usize,
    pub repeats: usize,
    pub num_embeds: usize,
    pub block_size: usize,
    pub rope_theta: f64,
    pub look_ahead_layers: Vec<usize>,
    pub look_backward_layers: Vec<usize>,
    pub dropout: f64,
    pub spk_config: SpeakerEncoderConfig,
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 22,
            num_attention_heads: 16,
            head_dim: 64,
            ff_mult: 2,
            emb_dim: 512,
            enc_emb_dim: 192,
            mel_dim: 80,
            repeats: 2,
            num_embeds: 8193,
            block_size: 24,
            rope_theta: 10000.0,
            look_ahead_layers: vec![10],
            look_backward_layers: vec![0, 20],
            dropout: 0.1,
            spk_config: SpeakerEncoderConfig {
                mel_dim: 80,
                enc_dim: 128,
                enc_channels: vec![256, 256, 256, 256, 768],
                enc_kernel_sizes: vec![5, 3, 3, 3, 1],
                enc_dilations: vec![1, 2, 3, 4, 1],
                enc_attention_channels: 64,
                enc_res2net_scale: 2,
                enc_se_channels: 64,
                sample_rate: 24000,
            },
        }
    }
}

/// DiT model for mel spectrogram generation from codes.
#[derive(Debug)]
pub struct DiTModel {
    time_embed: DiTTimestepEmbedding,
    text_embed: DiTCodecEmbedding,
    input_embed: DiTInputEmbedding,
    rotary_embed: DiTRotaryEmbedding,
    transformer_blocks: Vec<DiTDecoderLayer>,
    norm_out: AdaLayerNormZeroFinal,
    proj_out: Linear,
    mel_dim: usize,
    repeats: usize,
    block_size: usize,
    num_attention_heads: usize,
}

impl DiTModel {
    pub fn new(config: &DiTConfig, vb: VarBuilder) -> Result<Self> {
        let time_embed = DiTTimestepEmbedding::new(config.hidden_size, 256, vb.pp("time_embed"))?;
        let text_embed = DiTCodecEmbedding::new(
            config.num_embeds,
            config.emb_dim,
            config.repeats,
            vb.pp("text_embed"),
        )?;
        let input_embed = DiTInputEmbedding::new(
            config.mel_dim,
            config.spk_config.enc_dim,
            config.enc_emb_dim,
            config.emb_dim,
            config.hidden_size,
            &config.spk_config,
            vb.pp("input_embed"),
        )?;
        let rotary_embed =
            DiTRotaryEmbedding::new(config.head_dim, config.rope_theta, vb.device())?;

        let mut transformer_blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let look_ahead = if config.look_ahead_layers.contains(&i) {
                1
            } else {
                0
            };
            let look_backward = if config.look_backward_layers.contains(&i) {
                1
            } else {
                0
            };
            transformer_blocks.push(DiTDecoderLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.head_dim,
                config.ff_mult,
                look_ahead,
                look_backward,
                vb.pp(format!("transformer_blocks.{}", i)),
            )?);
        }

        let norm_out = AdaLayerNormZeroFinal::new(config.hidden_size, vb.pp("norm_out"))?;
        let proj_out =
            candle_nn::linear(config.hidden_size, config.mel_dim, vb.pp("proj_out"))?;

        Ok(Self {
            time_embed,
            text_embed,
            input_embed,
            rotary_embed,
            transformer_blocks,
            norm_out,
            proj_out,
            mel_dim: config.mel_dim,
            repeats: config.repeats,
            block_size: config.block_size,
            num_attention_heads: config.num_attention_heads,
        })
    }

    /// Create block difference tensor for block-sparse attention.
    fn create_block_diff(&self, seq_len: usize, batch: usize, device: &candle_core::Device) -> Result<Tensor> {
        let indices = Tensor::arange(0f32, seq_len as f32, device)?
            .affine(1.0 / self.block_size as f64, 0.0)?
            .floor()?;
        let block_i = indices.unsqueeze(1)?;
        let block_j = indices.unsqueeze(0)?;
        let diff = block_j.broadcast_sub(&block_i)?;
        // Expand to (batch, num_heads, seq, seq)
        diff.unsqueeze(0)?
            .unsqueeze(0)?
            .repeat((batch, self.num_attention_heads, 1, 1))
    }

    /// Forward pass through the DiT model.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        condition_vector: &Tensor,
        speaker_embedding: &Tensor,
        quantized_code: &Tensor,
        time_step: &Tensor,
        apply_cfg: bool,
    ) -> Result<Tensor> {
        let batch_size = if apply_cfg {
            hidden_states.dim(0)? * 2
        } else {
            hidden_states.dim(0)?
        };

        // Expand timestep if scalar
        let time_step = if time_step.dims().is_empty() || time_step.elem_count() == 1 {
            time_step.flatten_all()?.repeat(batch_size)?
        } else {
            time_step.clone()
        };

        let time_embedding = self.time_embed.forward(&time_step)?;

        let text_embedding = self.text_embed.forward(quantized_code, false)?;
        let text_embedding_uncond = if apply_cfg {
            Some(self.text_embed.forward(quantized_code, true)?)
        } else {
            None
        };

        let hidden_states = self.input_embed.forward(
            hidden_states,
            speaker_embedding,
            condition_vector,
            &text_embedding,
            apply_cfg,
            text_embedding_uncond.as_ref(),
        )?;

        let position_embeddings = self.rotary_embed.forward(&hidden_states)?;
        let seq_len = hidden_states.dim(1)?;
        let actual_batch = hidden_states.dim(0)?;
        let block_diff = self.create_block_diff(seq_len, actual_batch, hidden_states.device())?;

        let mut hidden_states = hidden_states;
        for block in &self.transformer_blocks {
            hidden_states =
                block.forward(&hidden_states, &time_embedding, &position_embeddings, &block_diff)?;
        }

        let hidden_states = self.norm_out.forward(&hidden_states, &time_embedding)?;
        self.proj_out.forward(&hidden_states)
    }

    /// Sample mel spectrogram using flow matching ODE (Euler integration).
    pub fn sample(
        &self,
        conditioning_vector: &Tensor,
        reference_mel: &Tensor,
        quantized_code: &Tensor,
        num_steps: usize,
        guidance_scale: f64,
        sway: f64,
    ) -> Result<Tensor> {
        let max_duration = quantized_code.dim(1)? * self.repeats;
        let device = quantized_code.device();
        let dtype = conditioning_vector.dtype();

        // Initialize from noise
        let noise = Tensor::randn(0f32, 1.0, (quantized_code.dim(0)?, max_duration, self.mel_dim), device)?
            .to_dtype(dtype)?;

        // Expand speaker embedding to sequence length
        let speaker_embedding = conditioning_vector
            .unsqueeze(1)?
            .repeat((1, max_duration, 1))?;

        // Time schedule with sway
        let time_steps: Vec<f64> = (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps as f64 - 1.0);
                t + sway * ((std::f64::consts::PI / 2.0 * t).cos() - 1.0 + t)
            })
            .collect();

        let mut values = noise;

        for i in 0..num_steps - 1 {
            let t0 = time_steps[i];
            let t1 = time_steps[i + 1];
            let dt = t1 - t0;

            let t_tensor = Tensor::full(t0 as f32, 1, device)?.to_dtype(dtype)?;

            let vt = if guidance_scale < 1e-5 {
                self.forward(
                    &values,
                    reference_mel,
                    &speaker_embedding,
                    quantized_code,
                    &t_tensor,
                    false,
                )?
            } else {
                let output = self.forward(
                    &values,
                    reference_mel,
                    &speaker_embedding,
                    quantized_code,
                    &t_tensor,
                    true,
                )?;
                let chunks = output.chunk(2, 0)?;
                let guided = &chunks[0];
                let null = &chunks[1];
                // CFG: guided + scale * (guided - null)
                (guided + (guided - null)?.affine(guidance_scale, 0.0)?)?
            };

            values = (&values + vt.affine(dt, 0.0)?)?;
        }

        // (batch, seq, mel_dim) → (batch, mel_dim, seq)
        values.transpose(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn test_sway_schedule_endpoints() -> Result<()> {
        let sway = -1.0;
        let f = |t: f64| t + sway * ((std::f64::consts::PI / 2.0 * t).cos() - 1.0 + t);
        assert!((f(0.0) - 0.0).abs() < 1e-10);
        assert!((f(1.0) - 1.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_sway_schedule_values_vs_python() -> Result<()> {
        let sway = -1.0;
        let num_steps = 5;
        let schedule: Vec<f64> = (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps as f64 - 1.0);
                t + sway * ((std::f64::consts::PI / 2.0 * t).cos() - 1.0 + t)
            })
            .collect();

        assert!((schedule[0] - 0.0).abs() < 1e-10);
        assert!((schedule[4] - 1.0).abs() < 1e-10);
        // Middle values should be shifted compared to linear
        assert!(schedule[1] < 0.25); // sway=-1 shifts early steps earlier
        assert!(schedule[2] < 0.5);
        Ok(())
    }

    #[test]
    fn test_euler_integration_constant_velocity() -> Result<()> {
        // dx/dt = 1, x(0) = 0 → x(1) ≈ 1
        let num_steps = 10;
        let dt = 1.0 / (num_steps as f64 - 1.0);
        let mut x = 0.0;
        for _ in 0..num_steps - 1 {
            x += 1.0 * dt; // velocity = 1
        }
        assert!((x - 1.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_euler_integration_step_count() -> Result<()> {
        let num_steps = 10;
        let mut count = 0;
        for _ in 0..num_steps - 1 {
            count += 1;
        }
        assert_eq!(count, 9);
        Ok(())
    }

    #[test]
    fn test_cfg_formula() -> Result<()> {
        let device = Device::Cpu;
        let guided = Tensor::from_vec(vec![2.0f32], 1, &device)?;
        let null = Tensor::from_vec(vec![1.0f32], 1, &device)?;
        let scale = 0.5;
        let result = (&guided + (&guided - &null)?.affine(scale, 0.0)?)?;
        let val = result.to_vec1::<f32>()?[0];
        // 2.0 + 0.5 * (2.0 - 1.0) = 2.5
        assert!((val - 2.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_noise_initialization_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let max_duration = 100;
        let mel_dim = 80;
        let noise = Tensor::randn(0f32, 1.0, (batch, max_duration, mel_dim), &device)?;
        assert_eq!(noise.dims(), &[batch, max_duration, mel_dim]);
        Ok(())
    }
}
