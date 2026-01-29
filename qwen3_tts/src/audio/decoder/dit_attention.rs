//! DiT attention module with block-sparse masking.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::audio::decoder::dit_rope::apply_rotary_pos_emb_dit;

/// Multi-head attention for DiT decoder layers.
#[derive(Debug)]
pub struct DiTAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl DiTAttention {
    pub fn new(hidden_size: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = num_heads * head_dim;
        let to_q = candle_nn::linear(hidden_size, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(hidden_size, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(hidden_size, inner_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(inner_dim, hidden_size, vb.pp("to_out.0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: &(Tensor, Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(hidden_states)?;
        let v = self.to_v.forward(hidden_states)?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE to all heads
        let (cos, sin) = position_embeddings;
        let (q, k) = apply_rotary_pos_emb_dit(&q, &k, cos, sin)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(1.0 / scale, 0.0)?;

        // Apply mask: true → attend (0.0), false → block (-inf)
        let attn_weights = if let Some(mask) = attention_mask {
            let mask_float = mask.to_dtype(DType::F32)?;
            // mask is bool-like: 1 = attend, 0 = block
            // Convert: 0 → -inf, 1 → 0
            let neg_inf = Tensor::full(f32::NEG_INFINITY, mask_float.shape(), mask_float.device())?;
            let zero = Tensor::zeros_like(&mask_float)?;
            let mask_u32 = mask.to_dtype(DType::U32)?;
            let additive_mask = mask_u32.where_cond(&zero, &neg_inf)?;
            (attn_weights.to_dtype(DType::F32)? + additive_mask)?.to_dtype(attn_weights.dtype())?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: (batch, heads, seq, head_dim) → (batch, seq, inner_dim)
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.to_out.forward(&attn_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    use crate::audio::decoder::dit_rope::DiTRotaryEmbedding;

    fn init_linear(vb: &VarBuilder, prefix: &str, in_d: usize, out_d: usize) -> Result<()> {
        let _ = vb
            .pp(prefix)
            .get_with_hints((out_d, in_d), "weight", candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 })?;
        let _ = vb
            .pp(prefix)
            .get_with_hints(out_d, "bias", candle_nn::Init::Const(0.0))?;
        Ok(())
    }

    #[test]
    fn test_dit_attention_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let hidden = 1024;
        let heads = 16;
        let head_dim = 64;
        let inner = heads * head_dim;

        init_linear(&vb, "to_q", hidden, inner)?;
        init_linear(&vb, "to_k", hidden, inner)?;
        init_linear(&vb, "to_v", hidden, inner)?;
        init_linear(&vb, "to_out.0", inner, hidden)?;

        let attn = DiTAttention::new(hidden, heads, head_dim, vb)?;

        let x = Tensor::randn(0f32, 1.0, (1, 20, hidden), &device)?;
        let rope = DiTRotaryEmbedding::new(head_dim, 10000.0, &device)?;
        let pos = rope.forward(&x)?;
        let output = attn.forward(&x, &pos, None)?;
        assert_eq!(output.dims(), &[1, 20, hidden]);
        Ok(())
    }

    #[test]
    fn test_attention_mask_bool_to_float() -> Result<()> {
        let device = Device::Cpu;
        let mask = Tensor::from_vec(vec![1u32, 0, 1, 0], (1, 1, 2, 2), &device)?;
        let zero = Tensor::zeros((1, 1, 2, 2), DType::F32, &device)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (1, 1, 2, 2), &device)?;
        let result = mask.where_cond(&zero, &neg_inf)?;
        let vals = result.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(vals[0], 0.0);
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        assert_eq!(vals[2], 0.0);
        assert!(vals[3].is_infinite() && vals[3] < 0.0);
        Ok(())
    }

    #[test]
    fn test_attention_with_identity_weights() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let hidden = 64;
        let heads = 4;
        let head_dim = 16;
        let inner = heads * head_dim;

        init_linear(&vb, "to_q", hidden, inner)?;
        init_linear(&vb, "to_k", hidden, inner)?;
        init_linear(&vb, "to_v", hidden, inner)?;
        init_linear(&vb, "to_out.0", inner, hidden)?;

        let attn = DiTAttention::new(hidden, heads, head_dim, vb)?;

        let x = Tensor::randn(0f32, 0.1, (1, 5, hidden), &device)?;
        let rope = DiTRotaryEmbedding::new(head_dim, 10000.0, &device)?;
        let pos = rope.forward(&x)?;
        let output = attn.forward(&x, &pos, None)?;

        // Check no NaN
        let has_nan = output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .any(|v| v.is_nan());
        assert!(!has_nan);
        Ok(())
    }
}
