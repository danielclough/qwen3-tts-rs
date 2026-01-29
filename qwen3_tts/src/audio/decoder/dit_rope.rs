//! DiT-specific Rotary Position Embeddings.
//!
//! Uses interleaved pairs instead of split-half rotation.

use candle_core::{DType, Device, Result, Tensor};

/// DiT rotary embedding with interleaved frequency pairs.
#[derive(Debug)]
pub struct DiTRotaryEmbedding {
    inv_freq: Tensor,
}

impl DiTRotaryEmbedding {
    pub fn new(dim: usize, base: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;
        Ok(Self { inv_freq })
    }

    /// Compute cos/sin embeddings for positions 0..seq_len.
    ///
    /// Returns (cos, sin) each of shape (batch, seq_len, dim) with interleaved pairs.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len) = (x.dim(0)?, x.dim(1)?);
        let device = x.device();

        let t = Tensor::arange(0f32, seq_len as f32, device)?;
        // freqs: (seq_len, dim/2) = t.unsqueeze(1) @ inv_freq.unsqueeze(0)
        let freqs = t
            .unsqueeze(1)?
            .to_dtype(DType::F32)?
            .matmul(&self.inv_freq.unsqueeze(0)?.to_dtype(DType::F32)?)?;

        // Interleave: stack (freqs, freqs) on last dim then reshape to (seq_len, dim)
        let freqs = Tensor::stack(&[&freqs, &freqs], candle_core::D::Minus1)?;
        let dim = self.inv_freq.dim(0)? * 2;
        let freqs = freqs.reshape((seq_len, dim))?;

        // Repeat for batch: (batch, seq_len, dim)
        let freqs = freqs.unsqueeze(0)?.repeat((batch_size, 1, 1))?;

        let cos = freqs.cos()?.to_dtype(x.dtype())?;
        let sin = freqs.sin()?.to_dtype(x.dtype())?;

        Ok((cos, sin))
    }
}

/// Rotate hidden dims using interleaved pairs: reshape to (..., d, 2), swap (-x2, x1), flatten.
pub fn rotate_half_codec(x: &Tensor) -> Result<Tensor> {
    let shape = x.dims().to_vec();
    let last = *shape.last().unwrap();
    // Reshape to (..., last/2, 2)
    let mut new_shape = shape[..shape.len() - 1].to_vec();
    new_shape.push(last / 2);
    new_shape.push(2);
    let x = x.reshape(new_shape.as_slice())?;

    let x1 = x.narrow(candle_core::D::Minus1, 0, 1)?;
    let x2 = x.narrow(candle_core::D::Minus1, 1, 1)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;

    // Flatten back
    let mut out_shape = new_shape[..new_shape.len() - 2].to_vec();
    out_shape.push(last);
    rotated.reshape(out_shape.as_slice())
}

/// Apply rotary position embedding (DiT variant with interleaved rotation).
pub fn apply_rotary_pos_emb_dit(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // q, k: (batch, heads, seq, head_dim)
    // cos, sin: (batch, seq, head_dim) -> unsqueeze(1) for heads
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;

    let q_embed = q
        .broadcast_mul(&cos)?
        .broadcast_add(&rotate_half_codec(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .broadcast_add(&rotate_half_codec(k)?.broadcast_mul(&sin)?)?;

    Ok((q_embed, k_embed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rotate_half_codec_interleaved_pairs() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (1, 1, 1, 8),
            &device,
        )?;
        let rotated = rotate_half_codec(&x)?;
        let vals = rotated.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(vals, vec![-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0]);
        Ok(())
    }

    #[test]
    fn test_rotate_half_codec_differs_from_standard() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (1, 1, 1, 8),
            &device,
        )?;
        let codec = rotate_half_codec(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let standard = crate::nn::rope::rotate_half(&x)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_ne!(codec, standard);
        Ok(())
    }

    #[test]
    fn test_dit_rope_cos_sin_shapes() -> Result<()> {
        let device = Device::Cpu;
        let rope = DiTRotaryEmbedding::new(64, 10000.0, &device)?;
        let x = Tensor::randn(0f32, 1.0, (2, 10, 64), &device)?;
        let (cos, sin) = rope.forward(&x)?;
        assert_eq!(cos.dims(), &[2, 10, 64]);
        assert_eq!(sin.dims(), &[2, 10, 64]);
        Ok(())
    }

    #[test]
    fn test_dit_rope_inv_freq_values() -> Result<()> {
        let device = Device::Cpu;
        let rope = DiTRotaryEmbedding::new(4, 10000.0, &device)?;
        let vals = rope.inv_freq.to_vec1::<f32>()?;
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 0.01).abs() < 1e-6);
        Ok(())
    }
}
