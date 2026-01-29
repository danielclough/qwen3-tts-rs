//! V1 (25Hz) tokenizer encoder.
//!
//! Wraps `WhisperEncoderVQ` — a Whisper encoder with group residual vector quantization.
//! Ref: `speech_vq.py:162-357`, `modeling_qwen3_tts_tokenizer_v1.py:1309-1340`

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder};

use super::whisper_v1::{WhisperV1Encoder, get_t_after_cnn};
use crate::audio::quantizer_v1::V1GroupResidualVectorQuantization;

/// Whisper encoder with vector quantization for the V1 tokenizer.
#[derive(Debug, Clone)]
pub struct TokenizerV1Encoder {
    encoder: WhisperV1Encoder,
    audio_vq_layers: usize,
    audio_vq_ds_rate: usize,
    audio_vq_downsample: Option<Conv1d>,
    audio_vq_upsample: Option<ConvTranspose1d>,
    audio_quantizer: V1GroupResidualVectorQuantization,
    audio_vq_pe: bool,
    project_after_vq_pe: Option<Linear>,
}

impl TokenizerV1Encoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_mels: usize,
        n_ctx: usize,
        n_state: usize,
        n_head: usize,
        n_layer: usize,
        n_window: usize,
        output_dim: usize,
        audio_vq_layers: usize,
        audio_vq_codebook_size: usize,
        audio_vq_codebook_dim: Option<usize>,
        audio_vq_pe: bool,
        audio_vq_ds_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let encoder = WhisperV1Encoder::new(
            n_mels, n_ctx, n_state, n_head, n_layer, n_window, output_dim, vb.clone(),
        )?;

        // The encoder's inherent ds_rate after CNN is 1 (since vq_layers > 0).
        // If the requested ds_rate differs, we need conv downsample/upsample.
        let inherent_ds_rate = 1usize;
        let (downsample, upsample) = if inherent_ds_rate == audio_vq_ds_rate {
            (None, None)
        } else {
            let stride = audio_vq_ds_rate / inherent_ds_rate;
            let ds = candle_nn::conv1d(
                n_state,
                n_state,
                stride,
                Conv1dConfig {
                    stride,
                    ..Default::default()
                },
                vb.pp("audio_vq_downsample"),
            )?;
            let us = candle_nn::conv_transpose1d(
                n_state,
                n_state,
                stride,
                ConvTranspose1dConfig {
                    stride,
                    ..Default::default()
                },
                vb.pp("audio_vq_upsample"),
            )?;
            (Some(ds), Some(us))
        };

        let audio_quantizer = V1GroupResidualVectorQuantization::new(
            1, // num_groups
            1, // num_quantizers
            n_state,
            audio_vq_codebook_size,
            audio_vq_codebook_dim,
            vb.pp("audio_quantizer"),
        )?;

        let project_after_vq_pe = if audio_vq_pe {
            Some(candle_nn::linear(n_state, n_state, vb.pp("project_after_vq_pe"))?)
        } else {
            None
        };

        Ok(Self {
            encoder,
            audio_vq_layers,
            audio_vq_ds_rate,
            audio_vq_downsample: downsample,
            audio_vq_upsample: upsample,
            audio_quantizer,
            audio_vq_pe,
            project_after_vq_pe,
        })
    }

    /// Quantize a packed tensor `(T, D)` with optional positional embedding.
    ///
    /// Returns `(quantized (T, D), indices (T/ds,))`.
    fn do_quantize(&self, x: &Tensor, pe: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        // (T, D) → (1, T, D)
        let x = x.unsqueeze(0)?;

        // Downsample: (1, T, D) → transpose to (1, D, T) → conv → transpose back
        let x = match &self.audio_vq_downsample {
            Some(ds) => {
                let xt = x.transpose(1, 2)?; // (1, D, T)
                let xt = ds.forward(&xt)?;
                xt.transpose(1, 2)? // (1, T', D)
            }
            None => x,
        };

        // Quantize
        let indices = self.audio_quantizer.encode(&x)?; // (n_q, num_groups, 1, T')
        let quantized = self.audio_quantizer.decode(&indices)?; // (1, T', D)

        // Squeeze indices: (n_q, num_groups, 1, T') → (T',)
        let indices = indices.squeeze(2)?.squeeze(1)?.squeeze(0)?;

        let mut x = quantized.squeeze(0)?; // (T', D)
        let indices_out = indices;

        // Add positional embedding if enabled
        if self.audio_vq_pe {
            if let Some(pe) = pe {
                x = (x + pe)?;
            }
            if let Some(proj) = &self.project_after_vq_pe {
                x = proj.forward(&x)?;
            }
        }

        // Upsample: (T', D) → (1, D, T') → conv_transpose → (T, D)
        let x = match &self.audio_vq_upsample {
            Some(us) => {
                let xt = x.unsqueeze(0)?.transpose(1, 2)?; // (1, D, T')
                let xt = us.forward(&xt)?;
                xt.transpose(1, 2)?.squeeze(0)? // (T, D)
            }
            None => x,
        };

        Ok((x, indices_out))
    }

    /// Forward pass returning `(encoded, indices)`.
    ///
    /// Runs through `audio_vq_layers` transformer blocks, then quantizes.
    /// This is the `return_indices=true` path from Python.
    pub fn forward_with_indices(
        &self,
        x_list: &[Tensor],
        audio_aftercnnlens: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let n_window = self.encoder.n_window;

        // Step 1-2: CNN + PE, also collect pe_for_vq
        let mut aftercnn_x_list = Vec::new();
        let mut pe_for_vq_list = Vec::new();

        for each_x in x_list {
            let total_frames = each_x.dim(1)?;
            let chunk_size = n_window * 2;
            let mut offset = 0;
            while offset < total_frames {
                let end = (offset + chunk_size).min(total_frames);
                let chunk = each_x.narrow(1, offset, end - offset)?;
                let processed = self.encoder.cnn_and_pe(&chunk)?;
                let chunk_len = processed.dim(0)?;

                // PE for VQ: positional_embedding[:chunk_len/ds_rate]
                let vq_len = chunk_len / self.audio_vq_ds_rate;
                let pe_vq = self.encoder.positional_embedding.narrow(0, 0, vq_len)?
                    .to_dtype(processed.dtype())?;
                pe_for_vq_list.push(pe_vq);

                aftercnn_x_list.push(processed);
                offset = end;
            }
        }

        let pe_for_vq = Tensor::cat(&pe_for_vq_list, 0)?;
        let mut x = Tensor::cat(&aftercnn_x_list, 0)?;

        // Compute cu_seqlens
        let cu_seqlens = WhisperV1Encoder::compute_cu_seqlens(audio_aftercnnlens, n_window);

        // Run blocks[0..audio_vq_layers]
        for block in self.encoder.blocks.iter().take(self.audio_vq_layers) {
            x = block.forward(&x, &cu_seqlens)?;
        }

        // Quantize
        self.do_quantize(&x, Some(&pe_for_vq))
    }

    /// Convert raw audio samples to mel spectrograms.
    ///
    /// Each audio is padded to a multiple of `160 * 2 * ds_rate` samples,
    /// then whisper-style log mel spectrogram (n_mels=128, n_fft=400, hop=160) is computed.
    pub fn speech2mel(&self, speechs: &[Vec<f32>], device: &Device) -> Result<Vec<Tensor>> {
        let mut mels = Vec::with_capacity(speechs.len());
        let reduction = 160 * 2 * self.audio_vq_ds_rate;

        for speech in speechs {
            let audio_len = speech.len();
            let padded_len = audio_len.div_ceil(reduction) * reduction;
            let mut padded = speech.clone();
            padded.resize(padded_len, 0.0);

            let audio_t = Tensor::from_vec(padded, padded_len, device)?;

            // Whisper-style mel: n_fft=400, hop=160, n_mels=128, sr=16000
            let mel = whisper_log_mel_spectrogram(&audio_t, 128, device)?;
            mels.push(mel);
        }
        Ok(mels)
    }

    /// Convert mel spectrograms to codes.
    ///
    /// Returns `(codes_per_audio, code_lens)`.
    pub fn mel2code(&self, mels: &[Tensor]) -> Result<(Vec<Tensor>, Vec<usize>)> {
        let audio_mellens: Vec<usize> = mels.iter().map(|m| m.dim(1)).collect::<Result<Vec<_>>>()?;
        let audio_aftercnnlens: Vec<usize> = audio_mellens.iter().map(|&t| get_t_after_cnn(t)).collect();

        let (_, indices) = self.forward_with_indices(mels, &audio_aftercnnlens)?;

        let indice_lens: Vec<usize> = audio_aftercnnlens
            .iter()
            .map(|&t| t / self.audio_vq_ds_rate)
            .collect();

        // Split indices by indice_lens
        let mut codes = Vec::with_capacity(indice_lens.len());
        let mut offset = 0;
        for &len in &indice_lens {
            codes.push(indices.narrow(0, offset, len)?);
            offset += len;
        }

        Ok((codes, indice_lens))
    }

    /// End-to-end: audio samples → codes.
    pub fn quantize_speech(
        &self,
        speechs: &[Vec<f32>],
        device: &Device,
    ) -> Result<(Vec<Tensor>, Vec<usize>)> {
        let mels = self.speech2mel(speechs, device)?;
        self.mel2code(&mels)
    }
}

/// Whisper-style log mel spectrogram (16kHz, n_fft=400, hop=160).
///
/// Matches `whisper_encoder.py:log_mel_spectrogram`.
fn whisper_log_mel_spectrogram(audio: &Tensor, n_mels: usize, device: &Device) -> Result<Tensor> {
    use crate::audio::mel::{create_hann_window, create_mel_filterbank, stft};

    let n_fft = 400;
    let hop_length = 160;
    let sample_rate = 16000;

    let samples = audio.to_vec1::<f32>()?;

    // PyTorch stft uses center=True by default, which pads n_fft//2 on each side
    let pad = n_fft / 2;
    let padded = crate::audio::mel::reflect_pad(&samples, pad, pad);

    let window = create_hann_window(n_fft);
    let stft_result = stft(&padded, n_fft, hop_length, n_fft, &window);

    let n_freqs = n_fft / 2 + 1;
    let n_frames = stft_result.len();

    if n_frames == 0 {
        return Tensor::zeros((n_mels, 0), DType::F32, device);
    }

    // Drop last frame to match PyTorch stft behavior: stft[..., :-1].abs() ** 2
    let n_frames_use = if n_frames > 0 { n_frames - 1 } else { 0 };

    // Magnitude squared
    let mut mag_data = vec![0.0f32; n_freqs * n_frames_use];
    for (frame_idx, frame) in stft_result[..n_frames_use].iter().enumerate() {
        for (freq_idx, c) in frame.iter().enumerate() {
            mag_data[freq_idx * n_frames_use + frame_idx] = c.re * c.re + c.im * c.im;
        }
    }
    let magnitudes = Tensor::from_vec(mag_data, (n_freqs, n_frames_use), device)?;

    // Mel filterbank
    let mel_fb = create_mel_filterbank(n_fft, n_mels, sample_rate, 0.0, Some(sample_rate as f64 / 2.0));
    let mel_fb_data: Vec<f32> = mel_fb.into_iter().flatten().collect();
    let mel_basis = Tensor::from_vec(mel_fb_data, (n_mels, n_freqs), device)?;

    let mel_spec = mel_basis.matmul(&magnitudes)?;

    // log10(clamp(x, 1e-10)), then max - 8.0 clamp, then (x + 4) / 4
    let log_spec = mel_spec.clamp(1e-10, f64::INFINITY)?;
    // log10(x) = ln(x) / ln(10)
    let log_spec = (log_spec.log()? * (1.0 / 10f64.ln()))?;
    let max_val = log_spec.max(D::Minus1)?.max(D::Minus1)?; // scalar
    let max_val = max_val.to_scalar::<f32>()?;
    let floor = max_val - 8.0;
    let log_spec = log_spec.clamp(floor as f64, f64::INFINITY)?;
    let log_spec = ((log_spec + 4.0)? / 4.0)?;

    Ok(log_spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_whisper_log_mel_shape() -> Result<()> {
        let audio = Tensor::zeros(16000, DType::F32, &Device::Cpu)?;
        let mel = whisper_log_mel_spectrogram(&audio, 128, &Device::Cpu)?;
        assert_eq!(mel.dim(0)?, 128);
        // center-padded: (16000 + 400 - 400) / 160 + 1 = 101, drop last = 100
        assert_eq!(mel.dim(1)?, 100);
        Ok(())
    }

    #[test]
    fn test_get_t_after_cnn_for_v1() {
        // Verify consistency with whisper_v1 module
        assert_eq!(get_t_after_cnn(100), 50);
        assert_eq!(get_t_after_cnn(3000), 1500);
    }
}
