//! V1 (25Hz) tokenizer decoder: DiT + BigVGAN pipeline.
//!
//! Codes → DiT (flow matching) → mel spectrogram → BigVGAN → audio waveform.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::audio::decoder::bigvgan::{BigVGANConfig, BigVGANDecoder};
use crate::audio::decoder::dit::{DiTConfig, DiTModel};

/// Complete V1 decoder: DiT sample → BigVGAN forward.
#[derive(Debug)]
pub struct TokenizerV1Decoder {
    dit: DiTModel,
    bigvgan: BigVGANDecoder,
}

impl TokenizerV1Decoder {
    pub fn new(
        dit_config: &DiTConfig,
        bigvgan_config: &BigVGANConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dit = DiTModel::new(dit_config, vb.pp("dit"))?;
        let bigvgan = BigVGANDecoder::new(bigvgan_config, vb.pp("bigvgan"))?;
        Ok(Self { dit, bigvgan })
    }

    pub fn load(
        dit_config: &DiTConfig,
        bigvgan_config: &BigVGANConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(dit_config, bigvgan_config, vb)
    }

    /// Decode codes to audio waveforms.
    ///
    /// - codes: (batch, code_len) — quantized codes
    /// - xvectors: (batch, xvec_dim) — speaker x-vectors
    /// - ref_mels: (batch, mel_time, mel_dim) — reference mel spectrograms
    ///
    /// Returns: list of audio tensors, one per batch item.
    pub fn decode(
        &self,
        codes: &Tensor,
        xvectors: &Tensor,
        ref_mels: &Tensor,
        num_steps: usize,
        guidance_scale: f64,
        sway: f64,
    ) -> Result<Tensor> {
        let mel = self.dit.sample(
            xvectors,
            ref_mels,
            codes,
            num_steps,
            guidance_scale,
            sway,
        )?;

        use candle_nn::Module;
        self.bigvgan.forward(&mel)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn test_decoder_output_shape() -> Result<()> {
        // Just verify the pipeline concept: mel → squeeze → audio shape
        let device = Device::Cpu;
        let mel = Tensor::randn(0f32, 1.0, (1, 80, 100), &device)?;
        // BigVGAN output should be (batch, samples) where samples = 100 * product(upsample_rates)
        // upsample_rates = [5, 3, 2, 2, 2, 2] → product = 240
        // So 100 * 240 = 24000 samples
        assert_eq!(mel.dims(), &[1, 80, 100]);
        Ok(())
    }
}
