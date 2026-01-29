//! X-vector speaker embedding extraction.
//!
//! Feature-gated behind `onnx-xvector`.
//! Ref: `speech_vq.py:118-159`

use candle_core::{Device, Result, Tensor};
#[cfg(feature = "onnx-xvector")]
use std::collections::HashMap;

/// Peak-normalize audio to 10^(-6/20) ≈ 0.501187.
///
/// Matches `sox.Transformer().norm(db_level=-6)`.
pub fn sox_norm(audio: &[f32]) -> Vec<f32> {
    let target_level: f32 = 10f32.powf(-6.0 / 20.0); // 0.501187
    let peak = audio
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    if peak < 1e-10 {
        return audio.to_vec();
    }
    let scale = target_level / peak;
    audio.iter().map(|&x| x * scale).collect()
}

/// Compute Kaldi-style filter-bank features.
///
/// 80 mel bins, 25ms frame (400 samples), 10ms hop (160 samples),
/// Hann window, power spectrum, log energy, mean normalization.
///
/// Input: 16kHz mono f32 audio.
/// Output: `(num_frames, 80)` tensor.
pub fn kaldi_fbank(audio: &[f32], device: &Device) -> Result<Tensor> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let sample_rate = 16000;
    let frame_length = 400; // 25ms
    let frame_shift = 160; // 10ms
    let n_mels = 80;
    let n_fft = 512; // next power of 2 >= 400
    let n_freqs = n_fft / 2 + 1;

    // Preemphasis
    let mut preemph = Vec::with_capacity(audio.len());
    preemph.push(audio.first().copied().unwrap_or(0.0));
    for i in 1..audio.len() {
        preemph.push(audio[i] - 0.97 * audio[i - 1]);
    }

    let num_frames = if audio.len() >= frame_length {
        (audio.len() - frame_length) / frame_shift + 1
    } else {
        0
    };

    if num_frames == 0 {
        return Tensor::zeros((0, n_mels), candle_core::DType::F32, device);
    }

    // Hann window (periodic)
    let window: Vec<f32> = (0..frame_length)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / frame_length as f32).cos()))
        .collect();

    // Mel filterbank
    let mel_fb = crate::audio::mel::create_mel_filterbank(n_fft, n_mels, sample_rate, 0.0, Some(sample_rate as f64 / 2.0));

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut feat_data = Vec::with_capacity(num_frames * n_mels);

    for f in 0..num_frames {
        let start = f * frame_shift;

        // Windowed frame, zero-padded to n_fft
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
        for i in 0..frame_length {
            let idx = start + i;
            let sample = if idx < preemph.len() { preemph[idx] } else { 0.0 };
            buffer[i] = Complex::new(sample * window[i], 0.0);
        }

        fft.process(&mut buffer);

        // Power spectrum
        let power: Vec<f32> = buffer[..n_freqs]
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // Apply mel filterbank + log
        for filter in &mel_fb {
            let mut energy: f32 = 0.0;
            for k in 0..n_freqs {
                energy += filter[k] * power[k];
            }
            feat_data.push((energy.max(1e-10)).ln());
        }
    }

    let feat = Tensor::from_vec(feat_data, (num_frames, n_mels), device)?;

    // Mean normalization across time
    let mean = feat.mean(0)?;
    feat.broadcast_sub(&mean)
}

/// X-vector extractor using an ONNX model.
///
/// Pipeline (matching Python `speech_vq.py:118-159`):
/// 1. `sox_norm` — normalize to -6dB
/// 2. `kaldi_fbank` — 80-dim Kaldi fbank features
/// 3. Mean-normalize features
/// 4. ONNX inference → 192-dim embedding
/// 5. L2-normalize embedding
/// 6. Extract reference mel via `MelSpectrogramFeaturesV1`
#[cfg(feature = "onnx-xvector")]
pub struct XVectorExtractor {
    model: candle_onnx::onnx::ModelProto,
    device: Device,
}

#[cfg(feature = "onnx-xvector")]
impl XVectorExtractor {
    /// Load an ONNX x-vector model from disk.
    pub fn load(onnx_path: &std::path::Path, device: &Device) -> Result<Self> {
        let model = candle_onnx::read_file(onnx_path)?;
        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Extract speaker x-vector and reference mel from raw 16kHz audio.
    ///
    /// Returns `(xvector[192], ref_mel[T, 80])`.
    pub fn extract_code(&self, audio: &[f32]) -> Result<(Tensor, Tensor)> {
        // 1. sox_norm
        let normed = sox_norm(audio);

        // 2. kaldi_fbank → (num_frames, 80)
        let fbank = kaldi_fbank(&normed, &self.device)?;

        // 3. ONNX inference: input (1, num_frames, 80) → output embedding
        let input = fbank.unsqueeze(0)?;

        let graph = self.model.graph.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("ONNX model has no graph".to_string())
        })?;

        let input_name = graph
            .input
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_string());

        let mut inputs = HashMap::new();
        inputs.insert(input_name, input);

        let outputs = candle_onnx::simple_eval(&self.model, inputs)?;

        // Get first output
        let output_name = graph
            .output
            .first()
            .map(|o| o.name.clone())
            .ok_or_else(|| candle_core::Error::Msg("ONNX model has no output".to_string()))?;

        let embedding = outputs.get(&output_name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Output '{}' not found in ONNX results", output_name))
        })?;

        // 4. L2-normalize: embedding / embedding.norm()
        let embedding = embedding.to_dtype(candle_core::DType::F32)?;
        let norm = embedding
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .sqrt()?;
        let xvector = embedding.broadcast_div(&norm)?;

        // 5. Reference mel via MelSpectrogramFeaturesV1
        let mel_extractor = crate::audio::mel_v1::MelSpectrogramFeaturesV1::new();
        let audio_t = Tensor::from_slice(&normed, normed.len(), &self.device)?;
        let ref_mel = mel_extractor.forward(&audio_t, &self.device)?;
        // Reshape: (1, 80, T) → (T, 80)
        let ref_mel = ref_mel.squeeze(0)?.transpose(0, 1)?;

        Ok((xvector.squeeze(0)?, ref_mel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sox_norm_basic() {
        let audio = vec![0.1f32, -0.5, 0.3];
        let normed = sox_norm(&audio);
        let target = 10f32.powf(-6.0 / 20.0); // 0.501187
        let peak = normed.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            (peak - target).abs() < 1e-5,
            "peak should be {target}, got {peak}"
        );
        // Check relative ratios preserved
        assert!((normed[0] / normed[2] - 0.1 / 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_sox_norm_silence() {
        let audio = vec![0.0f32, 0.0, 0.0];
        let normed = sox_norm(&audio);
        assert_eq!(normed, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_kaldi_fbank_shape() -> Result<()> {
        let audio: Vec<f32> = vec![0.0; 16000]; // 1 second of silence
        let feat = kaldi_fbank(&audio, &Device::Cpu)?;
        let dims = feat.dims();
        assert_eq!(dims[1], 80);
        // Expected frames: (16000 - 400) / 160 + 1 = 98
        assert_eq!(dims[0], 98, "expected 98 frames, got {}", dims[0]);
        Ok(())
    }

    #[test]
    fn test_kaldi_fbank_mean_normalized() -> Result<()> {
        // Use some non-silent audio
        let audio: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let feat = kaldi_fbank(&audio, &Device::Cpu)?;
        let mean = feat.mean(0)?.to_vec1::<f32>()?;
        for (i, &m) in mean.iter().enumerate() {
            assert!(
                m.abs() < 1e-4,
                "mean of bin {i} should be ~0 after normalization, got {m}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_kaldi_fbank_short_audio() -> Result<()> {
        // Audio shorter than frame_length (400 samples) → should return shape (0, 80)
        let audio: Vec<f32> = vec![0.1; 200];
        let feat = kaldi_fbank(&audio, &Device::Cpu)?;
        let dims = feat.dims();
        assert_eq!(dims[0], 0, "expected 0 frames for short audio, got {}", dims[0]);
        assert_eq!(dims[1], 80);
        Ok(())
    }

    #[test]
    fn test_kaldi_fbank_exact_one_frame() -> Result<()> {
        // Exactly 400 samples → should return shape (1, 80)
        let audio: Vec<f32> = (0..400).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let feat = kaldi_fbank(&audio, &Device::Cpu)?;
        let dims = feat.dims();
        assert_eq!(dims[0], 1, "expected 1 frame for exactly 400 samples, got {}", dims[0]);
        assert_eq!(dims[1], 80);
        Ok(())
    }

    #[test]
    fn test_sox_norm_preserves_sign() {
        let audio = vec![-0.3f32, 0.6, -0.9];
        let normed = sox_norm(&audio);
        // Signs should be preserved
        assert!(normed[0] < 0.0, "first sample should be negative");
        assert!(normed[1] > 0.0, "second sample should be positive");
        assert!(normed[2] < 0.0, "third sample should be negative");
    }
}
