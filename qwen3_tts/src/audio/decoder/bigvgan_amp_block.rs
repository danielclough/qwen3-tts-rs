//! BigVGAN AMPBlock: anti-aliased multi-period residual block.

use candle_core::Result;
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

use super::bigvgan_activations::{BigVGANCausalConv1d, TorchActivation1d};

// ---------------------------------------------------------------------------
// AMPBlock
// ---------------------------------------------------------------------------

/// Convs2 layer: either a regular Conv1d (causal_type="1") or causal (causal_type="2").
#[derive(Debug)]
pub enum Convs2Layer {
    Regular(Conv1d),
    Causal(BigVGANCausalConv1d),
}

impl Module for Convs2Layer {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        match self {
            Convs2Layer::Regular(conv) => conv.forward(xs),
            Convs2Layer::Causal(conv) => conv.forward(xs),
        }
    }
}

/// Anti-aliased multi-period residual block.
#[derive(Debug)]
pub struct AMPBlock {
    convs1: Vec<BigVGANCausalConv1d>,
    convs2: Vec<Convs2Layer>,
    activations: Vec<TorchActivation1d>,
    pre_conv: Option<Conv1d>,
    pre_act: Option<TorchActivation1d>,
}

impl AMPBlock {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        causal_type: &str,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let num_layers = dilations.len();

        let convs1: Vec<BigVGANCausalConv1d> = dilations
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                BigVGANCausalConv1d::new(channels, channels, kernel_size, 1, d, vb.pp(format!("convs1.{}", i)))
            })
            .collect::<Result<_>>()?;

        let padding = (kernel_size - 1) / 2;
        let convs2: Vec<Convs2Layer> = if causal_type == "1" {
            (0..num_layers)
                .map(|i| {
                    let config = Conv1dConfig {
                        padding,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                        ..Default::default()
                    };
                    let conv = candle_nn::conv1d(channels, channels, kernel_size, config, vb.pp(format!("convs2.{}", i)))?;
                    Ok(Convs2Layer::Regular(conv))
                })
                .collect::<Result<_>>()?
        } else {
            // causal_type "2": use BigVGANCausalConv1d for convs2
            (0..num_layers)
                .map(|i| {
                    let conv = BigVGANCausalConv1d::new(channels, channels, kernel_size, 1, 1, vb.pp(format!("convs2.{}", i)))?;
                    Ok(Convs2Layer::Causal(conv))
                })
                .collect::<Result<_>>()?
        };

        let total_acts = num_layers * 2;
        let activations: Vec<TorchActivation1d> = (0..total_acts)
            .map(|i| TorchActivation1d::new(channels, vb.pp(format!("activations.{}", i)), device))
            .collect::<Result<_>>()?;

        let (pre_conv, pre_act) = if causal_type == "2" {
            let config = Conv1dConfig {
                padding,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            let pc = candle_nn::conv1d(channels, channels, kernel_size, config, vb.pp("pre_conv"))?;
            let pa = TorchActivation1d::new(channels, vb.pp("pre_act"), device)?;
            (Some(pc), Some(pa))
        } else {
            (None, None)
        };

        Ok(Self {
            convs1,
            convs2,
            activations,
            pre_conv,
            pre_act,
        })
    }

    /// Returns true if this block has a pre_conv layer.
    pub fn has_pre_conv(&self) -> bool {
        self.pre_conv.is_some()
    }
}

impl Module for AMPBlock {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let mut hidden = if let (Some(pc), Some(pa)) = (&self.pre_conv, &self.pre_act) {
            let h = pc.forward(xs)?;
            pa.forward(&h)?
        } else {
            xs.clone()
        };

        let mut x = xs.clone();
        let acts1 = self.activations.iter().step_by(2);
        let acts2 = self.activations.iter().skip(1).step_by(2);

        for ((conv1, conv2), (act1, act2)) in self
            .convs1
            .iter()
            .zip(self.convs2.iter())
            .zip(acts1.zip(acts2))
        {
            hidden = act1.forward(&hidden)?;
            hidden = conv1.forward(&hidden)?;
            hidden = act2.forward(&hidden)?;
            hidden = conv2.forward(&hidden)?;
            x = (&x + &hidden)?;
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    fn init_amp_block(channels: usize, causal_type: &str) -> Result<AMPBlock> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let kernel_size = 3;
        let dilations = vec![1, 3, 5];
        let num_layers = dilations.len();

        // Init convs1 weights
        for (i, &_d) in dilations.iter().enumerate() {
            let w_vb = vb.pp(format!("convs1.{}", i));
            let _ = w_vb.get_with_hints((channels, channels, kernel_size), "weight", candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 })?;
            let _ = w_vb.get_with_hints(channels, "bias", candle_nn::Init::Const(0.0))?;
        }

        // Init convs2 weights
        for i in 0..num_layers {
            let w_vb = vb.pp(format!("convs2.{}", i));
            let _ = w_vb.get_with_hints((channels, channels, kernel_size), "weight", candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 })?;
            let _ = w_vb.get_with_hints(channels, "bias", candle_nn::Init::Const(0.0))?;
        }

        // Init activation weights
        for i in 0..(num_layers * 2) {
            let a_vb = vb.pp(format!("activations.{}", i));
            let _ = a_vb.get_with_hints(channels, "alpha", candle_nn::Init::Const(1.0))?;
            let _ = a_vb.get_with_hints(channels, "beta", candle_nn::Init::Const(1.0))?;
        }

        // Init pre_conv/pre_act for causal_type "2"
        if causal_type == "2" {
            let pc_vb = vb.pp("pre_conv");
            let _ = pc_vb.get_with_hints((channels, channels, kernel_size), "weight", candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 })?;
            let _ = pc_vb.get_with_hints(channels, "bias", candle_nn::Init::Const(0.0))?;
            let pa_vb = vb.pp("pre_act");
            let _ = pa_vb.get_with_hints(channels, "alpha", candle_nn::Init::Const(1.0))?;
            let _ = pa_vb.get_with_hints(channels, "beta", candle_nn::Init::Const(1.0))?;
        }

        AMPBlock::new(channels, kernel_size, &dilations, causal_type, vb, &device)
    }

    #[test]
    fn test_amp_block_residual_shape() -> Result<()> {
        let block = init_amp_block(256, "1")?;
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 256, 100), &device)?;
        let y = block.forward(&x)?;
        assert_eq!(y.dims(), &[1, 256, 100]);
        Ok(())
    }

    #[test]
    fn test_amp_block_causal_type_2_has_pre_conv() -> Result<()> {
        let block = init_amp_block(64, "2")?;
        assert!(block.has_pre_conv());
        Ok(())
    }

    #[test]
    fn test_amp_block_causal_type_1_no_pre_conv() -> Result<()> {
        let block = init_amp_block(64, "1")?;
        assert!(!block.has_pre_conv());
        Ok(())
    }
}
