//! V1 (25Hz) tokenizer: encoder + decoder pipeline.
//!
//! Reference: `modeling_qwen3_tts_tokenizer_v1.py:1360-1526`

pub mod config;

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use crate::audio::decoder::bigvgan::BigVGANConfig;
use crate::audio::decoder::dit::DiTConfig;
use crate::audio::decoder::v1::TokenizerV1Decoder;
use crate::audio::encoder::v1::TokenizerV1Encoder;
#[cfg(feature = "onnx-xvector")]
use crate::audio::encoder::xvector::XVectorExtractor;
use crate::audio::tokenizer::v1::config::TokenizerV1Config;
use crate::config::speaker_encoder_config::SpeakerEncoderConfig;

/// Model-level configuration extracted from `TokenizerV1Config`.
#[derive(Debug, Clone)]
pub struct TokenizerV1ModelConfig {
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
}

impl TokenizerV1ModelConfig {
    /// Tokenizer rate in Hz (always 25.0 for V1).
    pub fn tokenizer_rate(&self) -> f64 {
        25.0
    }
}

/// Output from the V1 encoder.
pub struct TokenizerV1EncoderOutput {
    /// Discrete codes `(batch, seq_len)`.
    pub audio_codes: Tensor,
    /// Speaker x-vectors `(batch, 192)`.
    pub xvectors: Tensor,
    /// Reference mel spectrograms `(batch, seq_len, 80)`.
    pub ref_mels: Tensor,
}

/// Complete V1 tokenizer combining encoder + decoder.
pub struct TokenizerV1 {
    encoder: Option<TokenizerV1Encoder>,
    decoder: TokenizerV1Decoder,
    #[cfg(feature = "onnx-xvector")]
    xvector_extractor: Option<XVectorExtractor>,
    config: TokenizerV1ModelConfig,
}

impl TokenizerV1 {
    /// Create a new V1 tokenizer.
    ///
    /// When `load_encoder` is false, only the decoder is loaded (for TTS generation).
    pub fn new(vb: VarBuilder, config: &TokenizerV1Config, load_encoder: bool) -> Result<Self> {
        let encoder = if load_encoder {
            Some(TokenizerV1Encoder::new(
                config.encoder_config.n_mels,
                config.encoder_config.n_ctx,
                config.encoder_config.n_state,
                config.encoder_config.n_head,
                config.encoder_config.n_layer,
                config.encoder_config.n_window,
                config.encoder_config.output_dim,
                config.encoder_config.audio_vq_layers,
                config.encoder_config.audio_vq_codebook_size,
                Some(config.encoder_config.audio_vq_codebook_dim),
                config.encoder_config.audio_vq_pe,
                config.encoder_config.audio_vq_ds_rate,
                vb.pp("encoder"),
            )?)
        } else {
            None
        };

        let dit_cfg = &config.decoder_config.dit_config;
        let dit_config = DiTConfig {
            hidden_size: dit_cfg.hidden_size,
            num_hidden_layers: dit_cfg.num_hidden_layers,
            num_attention_heads: dit_cfg.num_attention_heads,
            head_dim: dit_cfg.head_dim,
            ff_mult: dit_cfg.ff_mult,
            emb_dim: dit_cfg.emb_dim,
            enc_emb_dim: dit_cfg.enc_emb_dim,
            mel_dim: dit_cfg.mel_dim,
            repeats: dit_cfg.repeats,
            num_embeds: dit_cfg.num_embeds,
            block_size: dit_cfg.block_size,
            rope_theta: dit_cfg.rope_theta,
            look_ahead_layers: dit_cfg.look_ahead_layers.clone(),
            look_backward_layers: dit_cfg.look_backward_layers.clone(),
            dropout: dit_cfg.dropout,
            spk_config: SpeakerEncoderConfig {
                mel_dim: dit_cfg.mel_dim,
                enc_dim: dit_cfg.enc_dim,
                enc_channels: dit_cfg.enc_channels.clone(),
                enc_kernel_sizes: dit_cfg.enc_kernel_sizes.clone(),
                enc_dilations: dit_cfg.enc_dilations.clone(),
                enc_attention_channels: dit_cfg.enc_attention_channels,
                enc_res2net_scale: dit_cfg.enc_res2net_scale,
                enc_se_channels: dit_cfg.enc_se_channels,
                sample_rate: config.input_sample_rate,
            },
        };

        let bv_cfg = &config.decoder_config.bigvgan_config;
        let bigvgan_config = BigVGANConfig {
            mel_dim: bv_cfg.mel_dim,
            upsample_initial_channel: bv_cfg.upsample_initial_channel,
            resblock_kernel_sizes: bv_cfg.resblock_kernel_sizes.clone(),
            resblock_dilation_sizes: bv_cfg.resblock_dilation_sizes.clone(),
            upsample_rates: bv_cfg.upsample_rates.clone(),
            upsample_kernel_sizes: bv_cfg.upsample_kernel_sizes.clone(),
        };

        let decoder = TokenizerV1Decoder::new(&dit_config, &bigvgan_config, vb.pp("decoder"))?;

        let model_config = TokenizerV1ModelConfig {
            input_sample_rate: config.input_sample_rate,
            output_sample_rate: config.output_sample_rate,
            decode_upsample_rate: config.decode_upsample_rate,
            encode_downsample_rate: config.encode_downsample_rate,
        };

        Ok(Self {
            encoder,
            decoder,
            #[cfg(feature = "onnx-xvector")]
            xvector_extractor: None,
            config: model_config,
        })
    }

    /// Load the ONNX xvector extractor (encoder-side only).
    #[cfg(feature = "onnx-xvector")]
    pub fn load_xvector_extractor(&mut self, onnx_path: &std::path::Path) -> Result<()> {
        self.xvector_extractor = Some(XVectorExtractor::load(onnx_path, &candle_core::Device::Cpu)?);
        Ok(())
    }

    /// Encode audio to codes, x-vectors, and reference mels.
    ///
    /// Reference: `modeling_qwen3_tts_tokenizer_v1.py:1444-1485`
    #[allow(unused_variables)]
    pub fn encode(
        &mut self,
        audio: &Tensor,
        padding_mask: &Tensor,
    ) -> Result<TokenizerV1EncoderOutput> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Encoder not loaded".to_string()))?;
        let device = audio.device();
        let batch_size = audio.dim(0)?;

        // Trim each wav by padding mask (find last nonzero per batch)
        let mask_data = padding_mask.to_dtype(DType::F32)?;
        let mut wavs = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mask_row = mask_data.get(b)?.to_vec1::<f32>()?;
            let audio_row = audio.get(b)?.to_vec1::<f32>()?;
            let last_valid = mask_row
                .iter()
                .rposition(|&v| v > 0.0)
                .map(|i| i + 1)
                .unwrap_or(0);
            wavs.push(audio_row[..last_valid].to_vec());
        }

        // Quantize speech → codes
        let (codes_list, _codes_lens) = encoder.quantize_speech(&wavs, device)?;

        // Find max code length for padding
        let max_len = codes_list
            .iter()
            .map(|c| c.dim(0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);

        // Pad codes to max_len
        let mut padded_codes = Vec::with_capacity(batch_size);
        for codes in &codes_list {
            let len = codes.dim(0)?;
            if len < max_len {
                let padding = Tensor::zeros(max_len - len, DType::U32, device)?;
                padded_codes.push(Tensor::cat(&[codes, &padding], 0)?);
            } else {
                padded_codes.push(codes.clone());
            }
        }
        let audio_codes = Tensor::stack(&padded_codes, 0)?;

        // Extract x-vectors and reference mels
        #[cfg(feature = "onnx-xvector")]
        {
            let extractor = self
                .xvector_extractor
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("XVector extractor not loaded".to_string()))?;
            let mut xvec_list = Vec::with_capacity(batch_size);
            let mut mel_list = Vec::with_capacity(batch_size);
            for wav in &wavs {
                let (xvec, ref_mel) = extractor.extract_code(wav)?;
                xvec_list.push(xvec);
                mel_list.push(ref_mel);
            }
            let xvectors = Tensor::stack(&xvec_list, 0)?;
            // Pad ref_mels to max mel length
            let max_mel_len = mel_list
                .iter()
                .map(|m| m.dim(0))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            let mut padded_mels = Vec::with_capacity(batch_size);
            for mel in &mel_list {
                let len = mel.dim(0)?;
                if len < max_mel_len {
                    let padding = Tensor::zeros((max_mel_len - len, 80), DType::F32, device)?;
                    padded_mels.push(Tensor::cat(&[mel, &padding], 0)?);
                } else {
                    padded_mels.push(mel.clone());
                }
            }
            let ref_mels = Tensor::stack(&padded_mels, 0)?;
            Ok(TokenizerV1EncoderOutput {
                audio_codes,
                xvectors,
                ref_mels,
            })
        }
        #[cfg(not(feature = "onnx-xvector"))]
        {
            candle_core::bail!("XVector extraction requires the 'onnx-xvector' feature")
        }
    }

    /// Decode codes to audio waveforms with default sampling parameters.
    ///
    /// Default parameters: num_steps=10, guidance_scale=0.5, sway_coefficient=-1.0
    pub fn decode(
        &self,
        codes: &Tensor,
        xvectors: &Tensor,
        ref_mels: &Tensor,
    ) -> Result<Vec<Tensor>> {
        self.decode_with_params(codes, xvectors, ref_mels, 10, 0.5, -1.0)
    }

    /// Decode with explicit sampling parameters.
    ///
    /// Reference: `modeling_qwen3_tts_tokenizer_v1.py:1487-1525`
    pub fn decode_with_params(
        &self,
        codes: &Tensor,
        xvectors: &Tensor,
        ref_mels: &Tensor,
        num_steps: usize,
        guidance_scale: f64,
        sway_coefficient: f64,
    ) -> Result<Vec<Tensor>> {
        let waveform = self.decoder.decode(
            codes,
            xvectors,
            ref_mels,
            num_steps,
            guidance_scale,
            sway_coefficient,
        )?;

        let batch_size = codes.dim(0)?;
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let batch_codes = codes.get(b)?;
            // Count non-zero codes to determine audio length
            let codes_vec = batch_codes.to_vec1::<u32>()?;
            let codes_len = codes_vec.iter().filter(|&&c| c > 0).count();
            let audio_len = codes_len * self.config.decode_upsample_rate;

            let wav = waveform.get(b)?;
            let total_len = wav.dim(wav.dims().len() - 1)?;
            let trim_len = audio_len.min(total_len);

            // Trim to expected length
            let trimmed = if wav.dims().len() == 1 {
                wav.narrow(0, 0, trim_len)?
            } else {
                wav.narrow(wav.dims().len() - 1, 0, trim_len)?
            };
            results.push(trimmed);
        }

        Ok(results)
    }

    pub fn config(&self) -> &TokenizerV1ModelConfig {
        &self.config
    }

    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    pub fn input_sample_rate(&self) -> usize {
        self.config.input_sample_rate
    }

    pub fn output_sample_rate(&self) -> usize {
        self.config.output_sample_rate
    }

    pub fn encode_downsample_rate(&self) -> usize {
        self.config.encode_downsample_rate
    }

    pub fn decode_upsample_rate(&self) -> usize {
        self.config.decode_upsample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_defaults() {
        let cfg = TokenizerV1ModelConfig {
            input_sample_rate: 24000,
            output_sample_rate: 24000,
            decode_upsample_rate: 1920,
            encode_downsample_rate: 1920,
        };
        assert_eq!(cfg.tokenizer_rate(), 25.0);
        assert_eq!(cfg.input_sample_rate, 24000);
        assert_eq!(cfg.decode_upsample_rate, 1920);
    }

    #[test]
    fn test_trim_length_calculation() {
        let codes_len: usize = 10;
        let upsample_rate: usize = 1920;
        assert_eq!(codes_len * upsample_rate, 19200);
    }
}
