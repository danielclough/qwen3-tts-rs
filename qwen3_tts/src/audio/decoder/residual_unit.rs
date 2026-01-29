use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::audio::tokenizer::{
    snake_beta::SnakeBeta,
    v2::causal_conv::{CausalConv1d, CausalConvTranspose1d},
};

/// Residual unit for the vocoder.
#[derive(Debug)]
pub struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl DecoderResidualUnit {
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let act1 = SnakeBeta::new(dim, vb.pp("act1"))?;
        let conv1 = CausalConv1d::new(dim, dim, 7, dilation, 1, 1, vb.pp("conv1"))?;
        let act2 = SnakeBeta::new(dim, vb.pp("act2"))?;
        let conv2 = CausalConv1d::new(dim, dim, 1, 1, 1, 1, vb.pp("conv2"))?;

        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }
}

impl Module for DecoderResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.act1.forward(xs)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self.act2.forward(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;
        residual + hidden
    }
}

/// Decoder block with upsampling and residual units.
#[derive(Debug)]
pub struct DecoderBlock {
    act: SnakeBeta,
    upsample: CausalConvTranspose1d,
    residual_units: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        upsample_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let act = SnakeBeta::new(in_dim, vb.pp("block.0"))?;
        let upsample = CausalConvTranspose1d::new(
            in_dim,
            out_dim,
            2 * upsample_rate,
            upsample_rate,
            vb.pp("block.1"),
        )?;

        let residual_units = [1, 3, 9]
            .iter()
            .enumerate()
            .map(|(i, &dilation)| {
                DecoderResidualUnit::new(out_dim, dilation, vb.pp(format!("block.{}", i + 2)))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            act,
            upsample,
            residual_units,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.act.forward(xs)?;
        let mut hidden = self.upsample.forward(&hidden)?;

        for unit in &self.residual_units {
            hidden = unit.forward(&hidden)?;
        }

        Ok(hidden)
    }
}
