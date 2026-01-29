//! BigVGAN activation primitives: Kaiser sinc filters, up/downsampling, causal conv.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

use crate::audio::tokenizer::snake_beta::SnakeBeta;

// ---------------------------------------------------------------------------
// Kaiser sinc filter
// ---------------------------------------------------------------------------

/// Generate a 1D Kaiser-windowed sinc filter.
pub fn kaiser_sinc_filter1d(
    cutoff: f64,
    half_width: f64,
    kernel_size: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let is_even = kernel_size.is_multiple_of(2);
    let half_size = kernel_size / 2;

    let delta_f = 4.0 * half_width;
    let attenuation = 2.285 * (half_size as f64 - 1.0) * std::f64::consts::PI * delta_f + 7.95;

    let beta = kaiser_beta(attenuation);

    // Kaiser window
    let kaiser: Vec<f32> = (0..kernel_size)
        .map(|n| {
            let x = 2.0 * n as f64 / (kernel_size as f64 - 1.0) - 1.0;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();
            bessel_i0(arg) / bessel_i0(beta)
        })
        .map(|v| v as f32)
        .collect();

    // Time indices
    let time_indices: Vec<f64> = if is_even {
        (0..kernel_size)
            .map(|i| i as f64 - half_size as f64 + 0.5)
            .collect()
    } else {
        (0..kernel_size)
            .map(|i| i as f64 - half_size as f64)
            .collect()
    };

    if cutoff == 0.0 {
        return Tensor::zeros((1, 1, kernel_size), DType::F32, device);
    }

    // Sinc filter
    let filter: Vec<f32> = time_indices
        .iter()
        .zip(kaiser.iter())
        .map(|(&t, &w)| {
            let sinc_val = if (2.0 * cutoff * t).abs() < 1e-10 {
                1.0
            } else {
                let arg = std::f64::consts::PI * 2.0 * cutoff * t;
                arg.sin() / arg
            };
            (2.0 * cutoff * sinc_val) as f32 * w
        })
        .collect();

    let sum: f32 = filter.iter().sum();
    let normalized: Vec<f32> = filter.iter().map(|v| v / sum).collect();

    Tensor::from_vec(normalized, (1, 1, kernel_size), device)
}

/// Compute Kaiser window beta from attenuation.
pub fn kaiser_beta(attenuation: f64) -> f64 {
    if attenuation > 50.0 {
        0.1102 * (attenuation - 8.7)
    } else if attenuation >= 21.0 {
        0.5842 * (attenuation - 21.0).powf(0.4) + 0.07886 * (attenuation - 21.0)
    } else {
        0.0
    }
}

/// Modified Bessel function of the first kind, order 0.
pub fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..50 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-16 {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// UpSample1d / DownSample1d
// ---------------------------------------------------------------------------

/// Anti-aliased upsampling with Kaiser sinc filter.
#[derive(Debug)]
pub struct UpSample1d {
    filter: Tensor,
    ratio: usize,
    stride: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}

impl UpSample1d {
    pub fn new(ratio: usize, kernel_size: Option<usize>, device: &candle_core::Device) -> Result<Self> {
        let kernel_size = kernel_size.unwrap_or_else(|| (6 * ratio / 2) * 2);
        let stride = ratio;
        let pad = kernel_size / ratio - 1;
        let pad_left = pad * stride + (kernel_size - stride) / 2;
        let pad_right = pad * stride + (kernel_size - stride).div_ceil(2);

        let filter = kaiser_sinc_filter1d(
            0.5 / ratio as f64,
            0.6 / ratio as f64,
            kernel_size,
            device,
        )?;

        Ok(Self {
            filter,
            ratio,
            stride,
            pad,
            pad_left,
            pad_right,
        })
    }
}

impl Module for UpSample1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let channels = xs.dim(1)?;
        let padded = pad_replicate_1d(xs, self.pad, self.pad)?;
        let filter = self.filter.expand((channels, 1, self.filter.dim(2)?))?;
        let out = padded
            .conv_transpose1d(&filter.contiguous()?, 0, 0, self.stride, 1, channels)?
            .affine(self.ratio as f64, 0.0)?;
        let len = out.dim(2)?;
        out.narrow(2, self.pad_left, len - self.pad_left - self.pad_right)
    }
}

/// Anti-aliased downsampling with Kaiser sinc filter.
#[derive(Debug)]
pub struct DownSample1d {
    filter: Tensor,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
}

impl DownSample1d {
    pub fn new(ratio: usize, kernel_size: usize, device: &candle_core::Device) -> Result<Self> {
        let even = kernel_size.is_multiple_of(2);
        let pad_left = kernel_size / 2 - if even { 1 } else { 0 };
        let pad_right = kernel_size / 2;

        let filter = kaiser_sinc_filter1d(
            0.5 / ratio as f64,
            0.6 / ratio as f64,
            kernel_size,
            device,
        )?;

        Ok(Self {
            filter,
            stride: ratio,
            pad_left,
            pad_right,
        })
    }
}

impl Module for DownSample1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let channels = xs.dim(1)?;
        let padded = pad_replicate_1d(xs, self.pad_left, self.pad_right)?;
        let filter = self.filter.expand((channels, 1, self.filter.dim(2)?))?;
        padded.conv1d(&filter.contiguous()?, 0, self.stride, 1, channels)
    }
}

/// Replicate padding for 1D signals (batch, channels, time).
pub fn pad_replicate_1d(xs: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    if pad_left == 0 && pad_right == 0 {
        return Ok(xs.clone());
    }
    let length = xs.dim(2)?;
    let mut parts = Vec::new();
    if pad_left > 0 {
        let first = xs.narrow(2, 0, 1)?;
        parts.push(first.repeat((1, 1, pad_left))?);
    }
    parts.push(xs.clone());
    if pad_right > 0 {
        let last = xs.narrow(2, length - 1, 1)?;
        parts.push(last.repeat((1, 1, pad_right))?);
    }
    Tensor::cat(&parts.iter().collect::<Vec<_>>(), 2)
}

// ---------------------------------------------------------------------------
// TorchActivation1d
// ---------------------------------------------------------------------------

/// Upsample → activate → downsample for anti-aliased activation.
#[derive(Debug)]
pub struct TorchActivation1d {
    act: SnakeBeta,
    upsample: UpSample1d,
    downsample: DownSample1d,
}

impl TorchActivation1d {
    pub fn new(channels: usize, vb: VarBuilder, device: &candle_core::Device) -> Result<Self> {
        let act = SnakeBeta::new(channels, vb)?;
        let upsample = UpSample1d::new(2, Some(12), device)?;
        let downsample = DownSample1d::new(2, 12, device)?;
        Ok(Self {
            act,
            upsample,
            downsample,
        })
    }
}

impl Module for TorchActivation1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.upsample.forward(xs)?;
        let h = self.act.forward(&h)?;
        self.downsample.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// CausalConv1d (BigVGAN variant)
// ---------------------------------------------------------------------------

/// Causal convolution: left-pads by dilation*(kernel_size-1).
#[derive(Debug)]
pub struct BigVGANCausalConv1d {
    conv: Conv1d,
    causal_padding: usize,
}

impl BigVGANCausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let causal_padding = dilation * (kernel_size - 1);
        let config = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups: 1,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)?;
        Ok(Self {
            conv,
            causal_padding,
        })
    }
}

impl Module for BigVGANCausalConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let padded = xs.pad_with_zeros(2, self.causal_padding, 0)?;
        self.conv.forward(&padded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_kaiser_beta_high_attenuation() {
        let att = 60.0;
        let beta = kaiser_beta(att);
        let expected = 0.1102 * (60.0 - 8.7);
        assert!((beta - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kaiser_beta_medium_attenuation() {
        let att = 35.0;
        let beta = kaiser_beta(att);
        let expected = 0.5842 * (35.0_f64 - 21.0).powf(0.4) + 0.07886 * (35.0 - 21.0);
        assert!((beta - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kaiser_beta_low_attenuation() {
        let att = 15.0;
        let beta = kaiser_beta(att);
        assert_eq!(beta, 0.0);
    }

    #[test]
    fn test_kaiser_window_symmetry() {
        let len = 13;
        let beta = 5.0;
        let window: Vec<f64> = (0..len)
            .map(|n| {
                let x = 2.0 * n as f64 / (len as f64 - 1.0) - 1.0;
                let arg = beta * (1.0 - x * x).max(0.0).sqrt();
                bessel_i0(arg) / bessel_i0(beta)
            })
            .collect();
        for i in 0..len / 2 {
            assert!(
                (window[i] - window[len - 1 - i]).abs() < 1e-10,
                "Window not symmetric at index {}",
                i
            );
        }
    }

    #[test]
    fn test_kaiser_sinc_filter_shape() -> Result<()> {
        let device = Device::Cpu;
        let filter = kaiser_sinc_filter1d(0.25, 0.3, 12, &device)?;
        assert_eq!(filter.dims(), &[1, 1, 12]);
        Ok(())
    }

    #[test]
    fn test_kaiser_sinc_filter_normalized() -> Result<()> {
        let device = Device::Cpu;
        let filter = kaiser_sinc_filter1d(0.25, 0.3, 12, &device)?;
        let sum: f32 = filter.flatten_all()?.to_vec1::<f32>()?.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Filter sum = {}, expected ~1.0", sum);
        Ok(())
    }

    #[test]
    fn test_upsample1d_doubles_length() -> Result<()> {
        let device = Device::Cpu;
        let up = UpSample1d::new(2, Some(12), &device)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 10), &device)?;
        let y = up.forward(&x)?;
        assert_eq!(y.dims(), &[1, 4, 20]);
        Ok(())
    }

    #[test]
    fn test_downsample1d_halves_length() -> Result<()> {
        let device = Device::Cpu;
        let down = DownSample1d::new(2, 12, &device)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 20), &device)?;
        let y = down.forward(&x)?;
        assert_eq!(y.dims(), &[1, 4, 10]);
        Ok(())
    }

    #[test]
    fn test_upsample_then_downsample_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let up = UpSample1d::new(2, Some(12), &device)?;
        let down = DownSample1d::new(2, 12, &device)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 10), &device)?;
        let y = down.forward(&up.forward(&x)?)?;
        assert_eq!(y.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_causal_conv1d_output_length() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = vb.get_with_hints((16, 16, 3), "weight", candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 })?;
        let _ = vb.get_with_hints(16, "bias", candle_nn::Init::Const(0.0))?;
        let conv = BigVGANCausalConv1d::new(16, 16, 3, 1, 1, vb)?;
        let x = Tensor::randn(0f32, 1.0, (1, 16, 50), &device)?;
        let y = conv.forward(&x)?;
        assert_eq!(y.dim(2)?, 50);
        Ok(())
    }

    #[test]
    fn test_causal_conv1d_dilation_padding() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = vb.get_with_hints((16, 16, 3), "weight", candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 })?;
        let _ = vb.get_with_hints(16, "bias", candle_nn::Init::Const(0.0))?;
        let conv = BigVGANCausalConv1d::new(16, 16, 3, 1, 5, vb)?;
        assert_eq!(conv.causal_padding, 10);
        Ok(())
    }

    #[test]
    fn test_torch_activation1d_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = vb.get_with_hints(16, "alpha", candle_nn::Init::Const(1.0))?;
        let _ = vb.get_with_hints(16, "beta", candle_nn::Init::Const(1.0))?;
        let act = TorchActivation1d::new(16, vb, &device)?;
        let x = Tensor::randn(0f32, 1.0, (1, 16, 10), &device)?;
        let y = act.forward(&x)?;
        assert_eq!(y.dims(), &[1, 16, 10]);
        Ok(())
    }
}
