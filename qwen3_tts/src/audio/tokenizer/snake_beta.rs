//! SnakeBeta activation function.
//!
//! A modified Snake activation with separate alpha and beta parameters:
//! x + (1/beta) * sin²(x * alpha)

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// SnakeBeta activation function.
///
/// This activation function uses learnable parameters to control the
/// frequency (alpha) and magnitude (beta) of a periodic component added
/// to the input.
#[derive(Debug, Clone)]
pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
    eps: f64,
}

impl SnakeBeta {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(channels, "alpha")?;
        let beta = vb.get(channels, "beta")?;
        Ok(Self {
            alpha,
            beta,
            eps: 1e-9,
        })
    }

    pub fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(channels, vb)
    }
}

impl Module for SnakeBeta {
    /// Forward pass.
    ///
    /// Input shape: (batch, channels, time)
    /// Output shape: (batch, channels, time)
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();

        // Convert to F32 for precision in sin() computation
        // This is critical for audio quality - sin() of large values in BF16 loses precision
        let xs_f32 = xs.to_dtype(DType::F32)?;

        // alpha and beta are (channels,), need to reshape for broadcasting
        // Input is (batch, channels, time), so we need (1, channels, 1)
        let alpha = self
            .alpha
            .to_dtype(DType::F32)?
            .exp()?
            .unsqueeze(0)?
            .unsqueeze(2)?;
        let beta = self
            .beta
            .to_dtype(DType::F32)?
            .exp()?
            .unsqueeze(0)?
            .unsqueeze(2)?;

        // SnakeBeta(x) = x + (1/beta) * sin²(x * alpha)
        let sin_term = (xs_f32.broadcast_mul(&alpha)?).sin()?.sqr()?;
        let scale = sin_term.broadcast_div(&(beta + self.eps)?)?;

        let result = (xs_f32 + scale)?;

        // Convert back to original dtype
        result.to_dtype(original_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_snake_beta_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let channels = 64;
        let _ = vb.get_with_hints(channels, "alpha", candle_nn::Init::Const(0.0))?;
        let _ = vb.get_with_hints(channels, "beta", candle_nn::Init::Const(0.0))?;

        let snake = SnakeBeta::load(channels, vb)?;

        let input = Tensor::randn(0.0f32, 1.0, (2, channels, 100), &device)?;
        let output = snake.forward(&input)?;

        assert_eq!(output.dims(), input.dims());
        Ok(())
    }
}
