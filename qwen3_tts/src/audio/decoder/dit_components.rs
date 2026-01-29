//! DiT component modules: embeddings, norms, and feedforward.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

use crate::config::speaker_encoder_config::SpeakerEncoderConfig;
use crate::nn::speaker_encoder::SpeakerEncoder;

// ---------------------------------------------------------------------------
// DiTInputEmbedding
// ---------------------------------------------------------------------------

/// Projects concatenated mel + condition + code + speaker into hidden_size.
#[derive(Debug)]
pub struct DiTInputEmbedding {
    proj: Linear,
    spk_encoder: SpeakerEncoder,
}

impl DiTInputEmbedding {
    pub fn new(
        mel_dim: usize,
        enc_dim: usize,
        enc_emb_dim: usize,
        emb_dim: usize,
        hidden_size: usize,
        spk_config: &SpeakerEncoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let proj = candle_nn::linear(
            mel_dim + enc_dim + enc_emb_dim + emb_dim,
            hidden_size,
            vb.pp("proj"),
        )?;
        let spk_encoder = SpeakerEncoder::new(spk_config, vb.pp("spk_encoder"))?;
        Ok(Self { proj, spk_encoder })
    }

    /// Forward pass.
    ///
    /// - hidden_states: (batch, seq, mel_dim)
    /// - speaker_embedding: (batch, seq, enc_emb_dim) — pre-extracted x-vector repeated
    /// - condition_vector: (batch, mel_time, mel_dim) — reference mel
    /// - code_embed: (batch, seq, emb_dim)
    /// - apply_cfg: if true, doubles batch for classifier-free guidance
    /// - code_embed_uncond: unconditioned code embedding (required if apply_cfg)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        speaker_embedding: &Tensor,
        condition_vector: &Tensor,
        code_embed: &Tensor,
        apply_cfg: bool,
        code_embed_uncond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (hidden_states, speaker_embedding, condition_vector, code_embed) = if apply_cfg {
            let hs = Tensor::cat(&[hidden_states, hidden_states], 0)?;
            let spk = Tensor::cat(
                &[speaker_embedding, &Tensor::zeros_like(speaker_embedding)?],
                0,
            )?;
            let cv = Tensor::cat(
                &[condition_vector, &Tensor::zeros_like(condition_vector)?],
                0,
            )?;
            let ce = Tensor::cat(&[code_embed, code_embed_uncond.unwrap()], 0)?;
            (hs, spk, cv, ce)
        } else {
            (
                hidden_states.clone(),
                speaker_embedding.clone(),
                condition_vector.clone(),
                code_embed.clone(),
            )
        };

        // Run speaker encoder on condition_vector (reference mel)
        let spk_enc = self.spk_encoder.forward(&condition_vector)?; // (batch, enc_dim)
        let seq_len = hidden_states.dim(1)?;
        let spk_enc = spk_enc.unsqueeze(1)?.repeat((1, seq_len, 1))?; // (batch, seq, enc_dim)

        // Concatenate along feature dim and project
        let cat = Tensor::cat(
            &[&hidden_states, &spk_enc, &code_embed, &speaker_embedding],
            candle_core::D::Minus1,
        )?;
        self.proj.forward(&cat)
    }
}

// ---------------------------------------------------------------------------
// DiTCodecEmbedding
// ---------------------------------------------------------------------------

/// Embeds codec codes and repeats them along the time dimension.
#[derive(Debug)]
pub struct DiTCodecEmbedding {
    codec_embed: Embedding,
    repeats: usize,
}

impl DiTCodecEmbedding {
    pub fn new(num_embeds: usize, emb_dim: usize, repeats: usize, vb: VarBuilder) -> Result<Self> {
        // num_embeds + 1 for the padding/mask token
        let codec_embed = candle_nn::embedding(num_embeds + 1, emb_dim, vb.pp("codec_embed"))?;
        Ok(Self {
            codec_embed,
            repeats,
        })
    }

    pub fn forward(&self, codes: &Tensor, drop_code: bool) -> Result<Tensor> {
        let codes = if drop_code {
            Tensor::zeros_like(codes)?
        } else {
            codes.clone()
        };
        let embed = self.codec_embed.forward(&codes)?;
        // repeat_interleave along dim=1
        repeat_interleave(&embed, self.repeats, 1)
    }
}

/// Repeat each element along a dimension `repeats` times.
fn repeat_interleave(x: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    // Insert a new dim after `dim`, repeat, then flatten
    let shape = x.dims().to_vec();
    let mut new_shape = shape[..=dim].to_vec();
    new_shape.push(1);
    new_shape.extend_from_slice(&shape[dim + 1..]);
    let x = x.reshape(new_shape.as_slice())?;
    let mut repeat_dims = vec![1; new_shape.len()];
    repeat_dims[dim + 1] = repeats;
    let x = x.repeat(repeat_dims.as_slice())?;
    // Flatten dim and dim+1
    let mut out_shape = shape.clone();
    out_shape[dim] *= repeats;
    x.reshape(out_shape.as_slice())
}

// ---------------------------------------------------------------------------
// SinusPositionEmbedding
// ---------------------------------------------------------------------------

/// Sinusoidal position embedding for timesteps.
pub fn sinus_position_embedding(
    timestep: &Tensor,
    dim: usize,
    scale: f64,
) -> Result<Tensor> {
    let device = timestep.device();
    let half_dim = dim / 2;
    let emb_factor = -(10000f64.ln()) / (half_dim as f64 - 1.0);
    let emb = Tensor::arange(0f32, half_dim as f32, device)?
        .to_dtype(DType::F32)?
        .affine(emb_factor, 0.0)?
        .exp()?;
    // timestep: (batch,) -> (batch, 1) * emb: (half_dim,) -> (batch, half_dim)
    let emb = timestep
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_mul(&emb.unsqueeze(0)?)?
        .affine(scale, 0.0)?;
    let sin = emb.sin()?;
    let cos = emb.cos()?;
    let result = Tensor::cat(&[&sin, &cos], candle_core::D::Minus1)?;
    result.to_dtype(timestep.dtype())
}

// ---------------------------------------------------------------------------
// DiTTimestepEmbedding
// ---------------------------------------------------------------------------

/// Timestep embedding: sinusoidal → MLP.
#[derive(Debug)]
pub struct DiTTimestepEmbedding {
    freq_embed_dim: usize,
    linear1: Linear,
    linear2: Linear,
}

impl DiTTimestepEmbedding {
    pub fn new(dim: usize, freq_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(freq_embed_dim, dim, vb.pp("time_mlp.0"))?;
        let linear2 = candle_nn::linear(dim, dim, vb.pp("time_mlp.2"))?;
        Ok(Self {
            freq_embed_dim,
            linear1,
            linear2,
        })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let emb = sinus_position_embedding(timestep, self.freq_embed_dim, 1000.0)?;
        let emb = emb.to_dtype(timestep.dtype())?;
        let emb = self.linear1.forward(&emb)?.silu()?;
        self.linear2.forward(&emb)
    }
}

// ---------------------------------------------------------------------------
// AdaLayerNormZero
// ---------------------------------------------------------------------------

/// Adaptive LayerNorm with zero initialization.
///
/// Returns (normed_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp).
#[derive(Debug)]
pub struct AdaLayerNormZero {
    linear: Linear,
    norm_weight: Tensor,
    eps: f64,
}

impl AdaLayerNormZero {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(dim, dim * 6, vb.pp("linear"))?;
        let norm_weight = Tensor::ones(dim, DType::F32, vb.device())?;
        Ok(Self {
            linear,
            norm_weight,
            eps: 1e-6,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let emb = self.linear.forward(&emb.silu()?)?;
        let dim = emb.dim(candle_core::D::Minus1)? / 6;

        let shift_msa = emb.narrow(candle_core::D::Minus1, 0, dim)?;
        let scale_msa = emb.narrow(candle_core::D::Minus1, dim, dim)?;
        let gate_msa = emb.narrow(candle_core::D::Minus1, dim * 2, dim)?;
        let shift_mlp = emb.narrow(candle_core::D::Minus1, dim * 3, dim)?;
        let scale_mlp = emb.narrow(candle_core::D::Minus1, dim * 4, dim)?;
        let gate_mlp = emb.narrow(candle_core::D::Minus1, dim * 5, dim)?;

        // LayerNorm without affine
        let norm = LayerNorm::new_no_bias(
            self.norm_weight.to_dtype(hidden_states.dtype())?,
            self.eps,
        );
        let normed = norm.forward(hidden_states)?;

        // normed * (1 + scale_msa[:, None]) + shift_msa[:, None]
        let normed = normed
            .broadcast_mul(&scale_msa.unsqueeze(1)?.broadcast_add(&Tensor::ones(1, scale_msa.dtype(), scale_msa.device())?)?)?
            .broadcast_add(&shift_msa.unsqueeze(1)?)?;

        Ok((normed, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

// ---------------------------------------------------------------------------
// AdaLayerNormZeroFinal
// ---------------------------------------------------------------------------

/// Final adaptive LayerNorm (only scale + shift, no gate).
#[derive(Debug)]
pub struct AdaLayerNormZeroFinal {
    linear: Linear,
    norm_weight: Tensor,
    eps: f64,
}

impl AdaLayerNormZeroFinal {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(dim, dim * 2, vb.pp("linear"))?;
        let norm_weight = Tensor::ones(dim, DType::F32, vb.device())?;
        Ok(Self {
            linear,
            norm_weight,
            eps: 1e-6,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&emb.silu()?)?;
        let dim = emb.dim(candle_core::D::Minus1)? / 2;
        let scale = emb.narrow(candle_core::D::Minus1, 0, dim)?;
        let shift = emb.narrow(candle_core::D::Minus1, dim, dim)?;

        let norm = LayerNorm::new_no_bias(
            self.norm_weight.to_dtype(hidden_states.dtype())?,
            self.eps,
        );
        let normed = norm.forward(hidden_states)?;

        // normed * (1 + scale)[:, None, :] + shift[:, None, :]
        normed
            .broadcast_mul(
                &scale
                    .unsqueeze(1)?
                    .broadcast_add(&Tensor::ones(1, scale.dtype(), scale.device())?)?,
            )?
            .broadcast_add(&shift.unsqueeze(1)?)
    }
}

// ---------------------------------------------------------------------------
// DiTMLP
// ---------------------------------------------------------------------------

/// Feedforward MLP with GELU(tanh) activation.
#[derive(Debug)]
pub struct DiTMLP {
    fc1: Linear,
    fc2: Linear,
}

impl DiTMLP {
    pub fn new(dim: usize, ff_mult: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = dim * ff_mult;
        let fc1 = candle_nn::linear(dim, inner_dim, vb.pp("ff.0"))?;
        let fc2 = candle_nn::linear(inner_dim, dim, vb.pp("ff.3"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for DiTMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn test_vb(device: &Device) -> (VarMap, VarBuilder<'static>) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        (varmap, vb)
    }

    #[test]
    fn test_sinus_position_embedding_values() -> Result<()> {
        let device = Device::Cpu;
        let timestep = Tensor::zeros(1, DType::F32, &device)?;
        let emb = sinus_position_embedding(&timestep, 4, 1000.0)?;
        let vals = emb.flatten_all()?.to_vec1::<f32>()?;
        // sin(0) = 0, cos(0) = 1
        assert!((vals[0] - 0.0).abs() < 1e-6); // sin part
        assert!((vals[1] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 1.0).abs() < 1e-6); // cos part
        assert!((vals[3] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_sinus_position_embedding_timestep_1() -> Result<()> {
        let device = Device::Cpu;
        let timestep = Tensor::from_vec(vec![1f32], 1, &device)?;
        let emb = sinus_position_embedding(&timestep, 4, 1000.0)?;
        let vals = emb.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(vals.len(), 4);
        // sin(1000 * 1 * exp(0 * -ln(10000)/1)) = sin(1000)
        // sin(1000 * 1 * exp(1 * -ln(10000)/1)) = sin(0.1)
        let expected_sin_0 = (1000.0f32).sin();
        let expected_cos_0 = (1000.0f32).cos();
        assert!((vals[0] - expected_sin_0).abs() < 1e-4);
        assert!((vals[2] - expected_cos_0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_ada_layer_norm_zero_output_count() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let (_varmap, vb) = test_vb(&device);
        // Pre-init the linear weight/bias
        let _ = vb.pp("linear").get_with_hints(
            (dim * 6, dim),
            "weight",
            candle_nn::Init::Const(0.01),
        )?;
        let _ = vb.pp("linear").get_with_hints(
            dim * 6,
            "bias",
            candle_nn::Init::Const(0.0),
        )?;
        let norm = AdaLayerNormZero::new(dim, vb)?;

        let hidden = Tensor::randn(0f32, 1.0, (2, 10, dim), &device)?;
        let emb = Tensor::randn(0f32, 1.0, (2, dim), &device)?;
        let (normed, gate_msa, shift_mlp, scale_mlp, gate_mlp) = norm.forward(&hidden, &emb)?;

        assert_eq!(normed.dims(), &[2, 10, dim]);
        assert_eq!(gate_msa.dims(), &[2, dim]);
        assert_eq!(shift_mlp.dims(), &[2, dim]);
        assert_eq!(scale_mlp.dims(), &[2, dim]);
        assert_eq!(gate_mlp.dims(), &[2, dim]);
        Ok(())
    }

    #[test]
    fn test_ada_layer_norm_zero_final_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let dim = 64;
        let (_varmap, vb) = test_vb(&device);
        let _ = vb.pp("linear").get_with_hints(
            (dim * 2, dim),
            "weight",
            candle_nn::Init::Const(0.01),
        )?;
        let _ = vb.pp("linear").get_with_hints(
            dim * 2,
            "bias",
            candle_nn::Init::Const(0.0),
        )?;
        let norm = AdaLayerNormZeroFinal::new(dim, vb)?;

        let hidden = Tensor::randn(0f32, 1.0, (2, 10, dim), &device)?;
        let emb = Tensor::randn(0f32, 1.0, (2, dim), &device)?;
        let output = norm.forward(&hidden, &emb)?;
        assert_eq!(output.dims(), &[2, 10, dim]);
        Ok(())
    }

    #[test]
    fn test_dit_codec_embedding_repeat_interleave() -> Result<()> {
        let device = Device::Cpu;
        let (_varmap, vb) = test_vb(&device);
        let _ = vb.pp("codec_embed").get_with_hints(
            (4, 4),
            "weight",
            candle_nn::Init::Const(1.0),
        )?;
        let embed = DiTCodecEmbedding::new(3, 4, 2, vb)?;

        let codes = Tensor::from_vec(vec![0u32, 1, 2], (1, 3), &device)?;
        let output = embed.forward(&codes, false)?;
        assert_eq!(output.dims(), &[1, 6, 4]);
        Ok(())
    }

    #[test]
    fn test_dit_codec_embedding_drop_code() -> Result<()> {
        let device = Device::Cpu;
        let (_varmap, vb) = test_vb(&device);
        let _ = vb.pp("codec_embed").get_with_hints(
            (4, 4),
            "weight",
            candle_nn::Init::Randn { mean: 0.0, stdev: 1.0 },
        )?;
        let embed = DiTCodecEmbedding::new(3, 4, 2, vb)?;

        let codes = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device)?;
        let output = embed.forward(&codes, true)?;
        // When drop_code is true, codes become zeros, so all embeds should be same (embedding of 0)
        let first = output.narrow(1, 0, 1)?;
        let second = output.narrow(1, 2, 1)?;
        let diff = (first - second)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn test_dit_mlp_shapes() -> Result<()> {
        let device = Device::Cpu;
        let dim = 1024;
        let ff_mult = 2;
        let (_varmap, vb) = test_vb(&device);
        let _ = vb.pp("ff.0").get_with_hints(
            (dim * ff_mult, dim),
            "weight",
            candle_nn::Init::Const(0.01),
        )?;
        let _ = vb
            .pp("ff.0")
            .get_with_hints(dim * ff_mult, "bias", candle_nn::Init::Const(0.0))?;
        let _ = vb.pp("ff.3").get_with_hints(
            (dim, dim * ff_mult),
            "weight",
            candle_nn::Init::Const(0.01),
        )?;
        let _ = vb
            .pp("ff.3")
            .get_with_hints(dim, "bias", candle_nn::Init::Const(0.0))?;
        let mlp = DiTMLP::new(dim, ff_mult, vb)?;

        let input = Tensor::randn(0f32, 1.0, (2, 10, dim), &device)?;
        let output = mlp.forward(&input)?;
        assert_eq!(output.dims(), &[2, 10, dim]);
        Ok(())
    }
}
