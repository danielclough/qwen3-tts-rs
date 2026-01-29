//! Mel spectrogram computation for audio preprocessing.
//!
//! This module implements mel spectrogram extraction using STFT and mel filterbanks,
//! matching the Python reference implementation in modeling.py:399-464.

use candle_core::{Device, Result, Tensor};
use rustfft::{FftPlanner, num_complex::Complex};

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    /// FFT size
    pub n_fft: usize,
    /// Number of mel bins
    pub num_mels: usize,
    /// Audio sample rate
    pub sample_rate: usize,
    /// Hop size between frames
    pub hop_size: usize,
    /// Window size for STFT
    pub win_size: usize,
    /// Minimum frequency for mel filterbank
    pub fmin: f64,
    /// Maximum frequency for mel filterbank (None = sample_rate / 2)
    pub fmax: Option<f64>,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            num_mels: 128,
            sample_rate: 24000,
            hop_size: 256,
            win_size: 1024,
            fmin: 0.0,
            fmax: None,
        }
    }
}

/// Apply dynamic range compression (log compression).
///
/// Implements: log(clamp(x, clip_val) * C)
pub fn dynamic_range_compression(x: &Tensor, c: f64, clip_val: f64) -> Result<Tensor> {
    let clipped = x.clamp(clip_val, f64::INFINITY)?;
    (clipped * c)?.log()
}

/// Create a Hann window of given size.
///
/// Uses the periodic Hann window formula: 0.5 * (1 - cos(2*pi*n/N))
/// This matches torch.hann_window(N, periodic=True)
pub fn create_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            let x = 2.0 * std::f32::consts::PI * n as f32 / size as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// Compute mel spectrogram from audio waveform.
///
/// This implements the full mel spectrogram computation:
/// 1. Apply reflection padding
/// 2. Frame the audio with overlapping windows
/// 3. Apply Hann window
/// 4. Compute FFT
/// 5. Compute magnitude spectrum
/// 6. Apply mel filterbank
/// 7. Apply log compression
///
/// Args:
///   audio: Audio waveform of shape (batch, samples) or (samples,)
///   config: Mel spectrogram configuration
///   device: Device to compute on
///
/// Returns:
///   Mel spectrogram of shape (batch, num_mels, time)
pub fn mel_spectrogram(
    audio: &Tensor,
    config: &MelSpectrogramConfig,
    device: &Device,
) -> Result<Tensor> {
    // Ensure audio is 2D: (batch, samples)
    let audio = if audio.dims().len() == 1 {
        audio.unsqueeze(0)?
    } else {
        audio.clone()
    };

    let (batch_size, _num_samples) = audio.dims2()?;

    // Create mel filterbank matrix
    let mel_filterbank = create_mel_filterbank(
        config.n_fft,
        config.num_mels,
        config.sample_rate,
        config.fmin,
        config.fmax,
    );

    // Convert filterbank to tensor: (num_mels, n_fft/2 + 1)
    let n_freqs = config.n_fft / 2 + 1;
    let mel_basis_data: Vec<f32> = mel_filterbank.into_iter().flatten().collect();
    let mel_basis = Tensor::from_vec(mel_basis_data, (config.num_mels, n_freqs), device)?;

    // Create Hann window
    let hann_window = create_hann_window(config.win_size);

    // Padding for centering: (n_fft - hop_size) // 2
    let padding = (config.n_fft - config.hop_size) / 2;

    // Process each sample in batch
    let mut mel_specs = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let sample = audio.get(b)?.to_vec1::<f32>()?;

        // Apply reflection padding
        let padded = reflect_pad(&sample, padding, padding);

        // Compute STFT
        let stft_result = stft(
            &padded,
            config.n_fft,
            config.hop_size,
            config.win_size,
            &hann_window,
        );

        // Compute magnitude spectrum: sqrt(real^2 + imag^2 + 1e-9)
        let n_frames = stft_result.len();
        let mut magnitude_data = vec![0.0f32; n_freqs * n_frames];

        for (frame_idx, frame) in stft_result.iter().enumerate() {
            for (freq_idx, &complex_val) in frame.iter().enumerate() {
                let mag = (complex_val.re.powi(2) + complex_val.im.powi(2) + 1e-9).sqrt();
                magnitude_data[freq_idx * n_frames + frame_idx] = mag;
            }
        }

        // Create magnitude tensor: (n_freqs, n_frames)
        let magnitude = Tensor::from_vec(magnitude_data, (n_freqs, n_frames), device)?;

        // Apply mel filterbank: (num_mels, n_freqs) @ (n_freqs, n_frames) = (num_mels, n_frames)
        let mel_spec = mel_basis.matmul(&magnitude)?;

        mel_specs.push(mel_spec);
    }

    // Stack batch: (batch, num_mels, n_frames)
    let stacked = Tensor::stack(&mel_specs.iter().collect::<Vec<_>>(), 0)?;

    // Apply dynamic range compression: log(clamp(x, 1e-5) * 1.0)
    dynamic_range_compression(&stacked, 1.0, 1e-5)
}

/// Apply reflection padding to a 1D signal.
pub fn reflect_pad(signal: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let n = signal.len();
    let mut padded = Vec::with_capacity(n + pad_left + pad_right);

    // Left padding (reflect)
    for i in (1..=pad_left).rev() {
        let idx = if i < n { i } else { n - 1 };
        padded.push(signal[idx]);
    }

    // Original signal
    padded.extend_from_slice(signal);

    // Right padding (reflect)
    for i in 0..pad_right {
        let idx = if n >= 2 + i { n - 2 - i } else { 0 };
        padded.push(signal[idx]);
    }

    padded
}

/// Compute Short-Time Fourier Transform.
///
/// Args:
///   signal: Input signal (padded)
///   n_fft: FFT size
///   hop_size: Hop between frames
///   win_size: Window size
///   window: Hann window coefficients
///
/// Returns:
///   STFT result as Vec of frames, each frame is Vec of complex values (n_fft/2 + 1)
pub fn stft(
    signal: &[f32],
    n_fft: usize,
    hop_size: usize,
    win_size: usize,
    window: &[f32],
) -> Vec<Vec<Complex<f32>>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let n_freqs = n_fft / 2 + 1;
    let num_frames = (signal.len() - n_fft) / hop_size + 1;

    let mut result = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;

        // Create windowed frame (zero-padded if win_size < n_fft)
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];

        let offset = (n_fft - win_size) / 2;
        for i in 0..win_size {
            if start + i < signal.len() {
                buffer[offset + i] = Complex::new(signal[start + i] * window[i], 0.0);
            }
        }

        // Compute FFT in-place
        fft.process(&mut buffer);

        // Keep only positive frequencies (onesided)
        result.push(buffer[..n_freqs].to_vec());
    }

    result
}

/// Create mel filterbank matrix.
///
/// Creates a matrix that converts a power spectrum to a mel spectrum.
/// Uses the Slaney normalization (area normalization) to match librosa.
pub fn create_mel_filterbank(
    n_fft: usize,
    num_mels: usize,
    sample_rate: usize,
    fmin: f64,
    fmax: Option<f64>,
) -> Vec<Vec<f32>> {
    let fmax = fmax.unwrap_or(sample_rate as f64 / 2.0);
    let n_freqs = n_fft / 2 + 1;

    // Slaney-style mel scale (matches librosa default htk=False)
    // Linear below 1000 Hz, logarithmic above
    let f_sp = 200.0 / 3.0; // 66.67 Hz per mel below 1000 Hz
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // = 15
    let logstep = (6.4_f64).ln() / 27.0; // step size for log region

    let hz_to_mel = |hz: f64| {
        if hz < min_log_hz {
            hz / f_sp
        } else {
            min_log_mel + (hz / min_log_hz).ln() / logstep
        }
    };

    let mel_to_hz = |mel: f64| {
        if mel < min_log_mel {
            mel * f_sp
        } else {
            min_log_hz * ((mel - min_log_mel) * logstep).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Create mel points (num_mels + 2 points for triangular filters)
    let mel_points: Vec<f64> = (0..=num_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_mels + 1) as f64)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Create FFT frequency bins (in Hz)
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    // Create filterbank with triangular filters using float-based ramps (like librosa)
    let mut filterbank = vec![vec![0.0f32; n_freqs]; num_mels];

    for m in 0..num_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        // Rising slope: ramp from f_left to f_center
        // lower_slope = (freq - f_left) / (f_center - f_left)
        let lower_diff = f_center - f_left;

        // Falling slope: ramp from f_center to f_right
        // upper_slope = (f_right - freq) / (f_right - f_center)
        let upper_diff = f_right - f_center;

        for (bin_idx, &freq) in fft_freqs.iter().enumerate() {
            let lower = if lower_diff > 0.0 {
                (freq - f_left) / lower_diff
            } else {
                0.0
            };

            let upper = if upper_diff > 0.0 {
                (f_right - freq) / upper_diff
            } else {
                0.0
            };

            // Triangular filter: min of lower and upper ramps, clipped to 0
            let weight = lower.min(upper).max(0.0);
            filterbank[m][bin_idx] = weight as f32;
        }

        // Slaney normalization (area normalization)
        let enorm = 2.0 / (f_right - f_left) as f32;
        for val in &mut filterbank[m] {
            *val *= enorm;
        }
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = create_hann_window(4);
        assert_eq!(window.len(), 4);
        // Periodic Hann window: [0, 0.5, 1.0, 0.5] approximately
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reflect_pad() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&signal, 2, 2);
        // PyTorch reflect mode: [3, 2, | 1, 2, 3, 4, 5 | 4, 3]
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let filterbank = create_mel_filterbank(1024, 128, 24000, 0.0, Some(12000.0));
        assert_eq!(filterbank.len(), 128);
        assert_eq!(filterbank[0].len(), 513); // n_fft/2 + 1
    }

    #[test]
    fn test_mel_filterbank_values() {
        // Compare with Python librosa values
        let filterbank = create_mel_filterbank(1024, 128, 24000, 0.0, Some(12000.0));

        // Python librosa values:
        // Mel filterbank[0, :10]: [-0.0, 0.03355, 0.00857, 0.0, ...]
        // Mel filterbank[1, :10]: [0.0, 0.0, 0.02927, 0.01285, 0.0, ...]
        println!("Rust filterbank[0, :10]: {:?}", &filterbank[0][..10]);
        println!("Rust filterbank[1, :10]: {:?}", &filterbank[1][..10]);
        println!("Rust filterbank[64, :10]: {:?}", &filterbank[64][..10]);

        // Check first filter has values at bins 1-2
        assert!(filterbank[0][1] > 0.0, "filterbank[0][1] should be > 0");
        assert!(filterbank[0][2] > 0.0, "filterbank[0][2] should be > 0");
    }

    #[test]
    fn test_mel_spectrogram_shape() -> Result<()> {
        let device = Device::Cpu;
        let config = MelSpectrogramConfig {
            n_fft: 1024,
            num_mels: 128,
            sample_rate: 24000,
            hop_size: 256,
            win_size: 1024,
            fmin: 0.0,
            fmax: Some(12000.0),
        };

        // Create a simple test audio (1 second at 24kHz)
        let audio_data: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = Tensor::from_vec(audio_data, 24000, &device)?;

        let mel = mel_spectrogram(&audio, &config, &device)?;

        // Check output shape: (batch, num_mels, n_frames)
        let dims = mel.dims();
        assert_eq!(dims[0], 1); // batch
        assert_eq!(dims[1], 128); // num_mels

        Ok(())
    }
}
