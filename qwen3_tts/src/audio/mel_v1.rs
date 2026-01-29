//! V1-specific mel spectrogram computation.
//!
//! Two mel pipelines for the V1 tokenizer:
//! 1. Encoder mel (Whisper-style): Already in `encoder::v1::whisper_log_mel_spectrogram`
//! 2. BigVGAN mel (`MelSpectrogramFeaturesV1`): filter_length=1024, hop=160, win=640,
//!    n_mel=80, fmin=0, fmax=8000, sr=16000
//!
//! Reference: `speech_vq.py:42-115` (MelSpectrogramFeatures)

use candle_core::{Device, Result, Tensor};

use crate::audio::mel::{create_hann_window, create_mel_filterbank, reflect_pad, stft};

/// BigVGAN mel spectrogram feature extractor for V1 decoder.
///
/// Hardcoded parameters matching the Python reference:
/// - filter_length=1024, hop_length=160, win_length=640
/// - n_mel_channels=80, mel_fmin=0, mel_fmax=8000, sampling_rate=16000
///
/// Reference: `speech_vq.py:42-115`
#[derive(Debug, Clone)]
pub struct MelSpectrogramFeaturesV1 {
    pub filter_length: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mel_channels: usize,
    pub mel_fmin: f64,
    pub mel_fmax: f64,
    pub sampling_rate: usize,
}

impl Default for MelSpectrogramFeaturesV1 {
    fn default() -> Self {
        Self {
            filter_length: 1024,
            hop_length: 160,
            win_length: 640,
            n_mel_channels: 80,
            mel_fmin: 0.0,
            mel_fmax: 8000.0,
            sampling_rate: 16000,
        }
    }
}

impl MelSpectrogramFeaturesV1 {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute mel spectrogram from audio tensor.
    ///
    /// Input: audio tensor of shape `(samples,)` or `(batch, samples)`.
    /// Output: mel spectrogram of shape `(batch, n_mel_channels, time)`.
    ///
    /// Pipeline:
    /// 1. Reflect pad: `(filter_length - hop_length) / 2` on each side
    /// 2. STFT → complex → magnitude = sqrt(real^2 + imag^2 + 1e-9)
    /// 3. mel_spec = mel_basis @ magnitude
    /// 4. log(clamp(mel_spec, min=1e-5))  (dynamic_range_compression with C=1)
    pub fn forward(&self, audio: &Tensor, device: &Device) -> Result<Tensor> {
        let audio = if audio.dims().len() == 1 {
            audio.unsqueeze(0)?
        } else {
            audio.clone()
        };

        let (batch_size, _) = audio.dims2()?;

        let mel_fb = create_mel_filterbank(
            self.filter_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.mel_fmin,
            Some(self.mel_fmax),
        );
        let n_freqs = self.filter_length / 2 + 1;
        let mel_basis_data: Vec<f32> = mel_fb.into_iter().flatten().collect();
        let mel_basis = Tensor::from_vec(mel_basis_data, (self.n_mel_channels, n_freqs), device)?;

        let window = create_hann_window(self.win_length);
        let padding = (self.filter_length - self.hop_length) / 2;

        let mut mel_specs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let sample = audio.get(b)?.to_vec1::<f32>()?;
            let padded = reflect_pad(&sample, padding, padding);
            let stft_result = stft(
                &padded,
                self.filter_length,
                self.hop_length,
                self.win_length,
                &window,
            );

            let n_frames = stft_result.len();
            let mut magnitude_data = vec![0.0f32; n_freqs * n_frames];

            for (frame_idx, frame) in stft_result.iter().enumerate() {
                for (freq_idx, &c) in frame.iter().enumerate() {
                    let mag = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
                    magnitude_data[freq_idx * n_frames + frame_idx] = mag;
                }
            }

            let magnitude = Tensor::from_vec(magnitude_data, (n_freqs, n_frames), device)?;
            let mel_spec = mel_basis.matmul(&magnitude)?;
            mel_specs.push(mel_spec);
        }

        let stacked = Tensor::stack(&mel_specs.iter().collect::<Vec<_>>(), 0)?;
        crate::audio::mel::dynamic_range_compression(&stacked, 1.0, 1e-5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_v1_defaults() {
        let mel = MelSpectrogramFeaturesV1::new();
        assert_eq!(mel.filter_length, 1024);
        assert_eq!(mel.hop_length, 160);
        assert_eq!(mel.win_length, 640);
        assert_eq!(mel.n_mel_channels, 80);
        assert_eq!(mel.mel_fmin, 0.0);
        assert_eq!(mel.mel_fmax, 8000.0);
        assert_eq!(mel.sampling_rate, 16000);
    }

    #[test]
    fn test_mel_v1_shape() -> Result<()> {
        let mel = MelSpectrogramFeaturesV1::new();
        let device = Device::Cpu;
        // 1 second of 16kHz audio
        let audio_data: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = Tensor::from_vec(audio_data, 16000, &device)?;
        let result = mel.forward(&audio, &device)?;
        let dims = result.dims();
        assert_eq!(dims[0], 1); // batch
        assert_eq!(dims[1], 80); // n_mel_channels
        // frames = (16000 + 2*432 - 1024) / 160 + 1 = 100
        assert_eq!(dims[2], 100, "expected 100 frames, got {}", dims[2]);
        Ok(())
    }

    #[test]
    fn test_mel_v1_transpose_to_time_major() -> Result<()> {
        let mel = MelSpectrogramFeaturesV1::new();
        let device = Device::Cpu;
        // 1 second of 16kHz audio
        let audio_data: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = Tensor::from_vec(audio_data, 16000, &device)?;
        let result = mel.forward(&audio, &device)?;
        // Shape is (1, 80, T) — apply same reshape as extract_code
        let transposed = result.squeeze(0)?.transpose(0, 1)?;
        let dims = transposed.dims();
        assert_eq!(dims[0], 100, "expected 100 time frames, got {}", dims[0]);
        assert_eq!(dims[1], 80, "expected 80 mel channels, got {}", dims[1]);
        Ok(())
    }

    #[test]
    fn test_mel_v1_non_silent_range() -> Result<()> {
        let mel = MelSpectrogramFeaturesV1::new();
        let device = Device::Cpu;
        // 1 second of 440Hz sine wave at 16kHz
        let audio_data: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();
        let audio = Tensor::from_vec(audio_data, 16000, &device)?;
        let result = mel.forward(&audio, &device)?;
        let values = result.flatten_all()?.to_vec1::<f32>()?;
        // All values should be finite
        assert!(values.iter().all(|v| v.is_finite()), "all values should be finite");
        // Not all identical (real audio should produce varied mel values)
        let first = values[0];
        assert!(
            values.iter().any(|&v| (v - first).abs() > 1e-6),
            "mel values should not all be identical"
        );
        Ok(())
    }
}
