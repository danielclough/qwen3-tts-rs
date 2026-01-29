//! DiT decoder layer (transformer block with adaptive normalization).

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};

use crate::audio::decoder::dit_attention::DiTAttention;
use crate::audio::decoder::dit_components::{AdaLayerNormZero, DiTMLP};

/// Single DiT decoder block.
#[derive(Debug)]
pub struct DiTDecoderLayer {
    attn_norm: AdaLayerNormZero,
    attn: DiTAttention,
    look_ahead_block: i64,
    look_backward_block: i64,
    ff_norm_weight: Tensor,
    ff_norm_eps: f64,
    ff: DiTMLP,
}

impl DiTDecoderLayer {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        ff_mult: usize,
        look_ahead_block: i64,
        look_backward_block: i64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn_norm = AdaLayerNormZero::new(hidden_size, vb.pp("attn_norm"))?;
        let attn = DiTAttention::new(hidden_size, num_heads, head_dim, vb.pp("attn"))?;
        let ff_norm_weight =
            candle_core::Tensor::ones(hidden_size, candle_core::DType::F32, vb.device())?;
        let ff = DiTMLP::new(hidden_size, ff_mult, vb.pp("ff"))?;

        Ok(Self {
            attn_norm,
            attn,
            look_ahead_block,
            look_backward_block,
            ff_norm_weight,
            ff_norm_eps: 1e-6,
            ff,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        position_embeddings: &(Tensor, Tensor),
        block_diff: &Tensor,
    ) -> Result<Tensor> {
        // Pre-norm + modulation
        let (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.attn_norm.forward(hidden_states, timestep)?;

        // Block-sparse attention mask from block_diff
        // attend where: block_diff >= -look_backward AND block_diff <= look_ahead
        let lb = Tensor::full(
            -self.look_backward_block as f32,
            block_diff.shape(),
            block_diff.device(),
        )?;
        let la = Tensor::full(
            self.look_ahead_block as f32,
            block_diff.shape(),
            block_diff.device(),
        )?;
        let block_diff_f32 = block_diff.to_dtype(candle_core::DType::F32)?;
        let mask = block_diff_f32.ge(&lb)?.mul(&block_diff_f32.le(&la)?)?;

        // Attention
        let attn_output = self.attn.forward(&norm, position_embeddings, Some(&mask))?;

        // Residual + gated attention
        let hidden_states =
            (hidden_states + gate_msa.unsqueeze(1)?.broadcast_mul(&attn_output)?)?;

        // FF norm (no affine) + modulation
        let ff_norm = LayerNorm::new_no_bias(
            self.ff_norm_weight.to_dtype(hidden_states.dtype())?,
            self.ff_norm_eps,
        );
        let normed = ff_norm.forward(&hidden_states)?;
        let ones = Tensor::ones(1, scale_mlp.dtype(), scale_mlp.device())?;
        let normed = normed
            .broadcast_mul(&scale_mlp.unsqueeze(1)?.broadcast_add(&ones)?)?
            .broadcast_add(&shift_mlp.unsqueeze(1)?)?;

        // FF + gated residual
        let ff_output = self.ff.forward(&normed)?;
        &hidden_states + gate_mlp.unsqueeze(1)?.broadcast_mul(&ff_output)?
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn test_block_diff_values() -> Result<()> {
        let device = Device::Cpu;
        let seq_len = 48usize;
        let block_size = 24usize;

        let indices = Tensor::arange(0f32, seq_len as f32, &device)?
            .affine(1.0 / block_size as f64, 0.0)?
            .floor()?;
        let block_i = indices.unsqueeze(1)?;
        let block_j = indices.unsqueeze(0)?;
        let diff = block_j.broadcast_sub(&block_i)?;
        let vals = diff.flatten_all()?.to_vec1::<f32>()?;

        // First element in block 0 vs last element in block 1: diff = 1
        assert!((vals[47] - 1.0).abs() < 1e-6);
        // Same block: diff = 0
        assert!((vals[0] - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_attention_mask_from_block_diff() -> Result<()> {
        let device = Device::Cpu;
        let block_diff = Tensor::from_vec(vec![-2f32, -1., 0., 1., 2.], (1, 1, 1, 5), &device)?;

        let look_backward: f32 = 1.0;
        let look_ahead: f32 = 1.0;
        let lb = Tensor::full(-look_backward, block_diff.shape(), &device)?;
        let la = Tensor::full(look_ahead, block_diff.shape(), &device)?;
        let mask = block_diff.ge(&lb)?.mul(&block_diff.le(&la)?)?;
        let vals = mask.flatten_all()?.to_vec1::<u8>()?;
        // -2: no, -1: yes, 0: yes, 1: yes, 2: no
        assert_eq!(vals, vec![0, 1, 1, 1, 0]);
        Ok(())
    }

    #[test]
    fn test_block_diff_three_blocks() -> Result<()> {
        let device = Device::Cpu;
        let seq_len = 72usize;
        let block_size = 24usize;

        let indices = Tensor::arange(0f32, seq_len as f32, &device)?
            .affine(1.0 / block_size as f64, 0.0)?
            .floor()?;
        let block_i = indices.unsqueeze(1)?;
        let block_j = indices.unsqueeze(0)?;
        let diff = block_j.broadcast_sub(&block_i)?;

        let min = diff.flatten_all()?.min(0)?.to_scalar::<f32>()?;
        let max = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!((min - (-2.0)).abs() < 1e-6);
        assert!((max - 2.0).abs() < 1e-6);
        Ok(())
    }
}
