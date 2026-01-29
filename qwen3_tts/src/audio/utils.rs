//! Audio loading and processing utilities.
//!
//! This module provides utilities for loading audio from various sources:
//! - File paths (wav, mp3, flac, ogg, etc.)
//! - Base64-encoded audio strings
//! - URL fetching (future)
//!
//! # Feature Flag
//!
//! This module requires the `audio-loading` feature:
//!
//! ```toml
//! candle-qwen3-tts = { version = "0.1", features = ["audio-loading"] }
//! ```

use candle_core::{Device, Result, Tensor};

/// Result of loading audio.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Audio samples as f32 in range [-1, 1].
    pub samples: Vec<f32>,
    /// Sample rate of the audio.
    pub sample_rate: u32,
    /// Number of channels (1 for mono, 2 for stereo).
    pub channels: usize,
}

impl AudioData {
    /// Create new audio data.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: usize) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Convert to mono by averaging channels.
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let mono_samples: Vec<f32> = self
            .samples
            .chunks(self.channels)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect();

        Self {
            samples: mono_samples,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Convert to a Candle tensor.
    ///
    /// Returns a 1D tensor of shape `(samples,)` for mono audio,
    /// or 2D tensor of shape `(channels, samples)` for multi-channel.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        if self.channels == 1 {
            Tensor::from_vec(self.samples.clone(), (self.samples.len(),), device)
        } else {
            let samples_per_channel = self.samples.len() / self.channels;
            // Deinterleave: [L, R, L, R, ...] -> [[L, L, ...], [R, R, ...]]
            let mut channels_data = vec![Vec::with_capacity(samples_per_channel); self.channels];
            for (i, &sample) in self.samples.iter().enumerate() {
                channels_data[i % self.channels].push(sample);
            }
            let flat: Vec<f32> = channels_data.into_iter().flatten().collect();
            Tensor::from_vec(flat, (self.channels, samples_per_channel), device)
        }
    }

    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f32 {
        let samples_per_channel = self.samples.len() / self.channels;
        samples_per_channel as f32 / self.sample_rate as f32
    }
}

/// Audio input type that can be converted to audio data.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Path to an audio file.
    FilePath(String),
    /// Base64-encoded audio data.
    Base64(String),
    /// Raw audio samples with sample rate.
    Raw {
        samples: Vec<f32>,
        sample_rate: u32,
        channels: usize,
    },
    /// Pre-loaded audio data.
    AudioData(AudioData),
}

impl From<AudioData> for AudioInput {
    fn from(data: AudioData) -> Self {
        AudioInput::AudioData(data)
    }
}

impl From<String> for AudioInput {
    fn from(s: String) -> Self {
        if is_probably_base64(&s) {
            AudioInput::Base64(s)
        } else {
            AudioInput::FilePath(s)
        }
    }
}

impl From<&str> for AudioInput {
    fn from(s: &str) -> Self {
        AudioInput::from(s.to_string())
    }
}

/// Check if a string is probably base64-encoded audio.
///
/// Detection heuristics:
/// 1. Starts with "data:audio" (data URL format)
/// 2. No path separators and very long (>256 chars)
pub fn is_probably_base64(s: &str) -> bool {
    // Data URL format
    if s.starts_with("data:audio") {
        return true;
    }

    // Heuristic: no path separators and long string
    if !s.contains('/') && !s.contains('\\') && s.len() > 256 {
        // Additional check: should be valid base64 characters
        let base64_chars = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '+' || *c == '/' || *c == '=')
            .count();
        // If >90% of chars are base64-valid, likely base64
        return base64_chars as f32 / s.len() as f32 > 0.9;
    }

    false
}

/// Check if a string is a URL.
pub fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// Decode base64 audio data, handling data URL format.
#[cfg(feature = "audio-loading")]
pub fn decode_base64_to_bytes(b64: &str) -> Result<Vec<u8>> {
    use base64::Engine;

    let data = if b64.contains(',') && b64.trim().starts_with("data:") {
        // Data URL format: "data:audio/wav;base64,XXXX..."
        b64.split(',')
            .nth(1)
            .ok_or_else(|| candle_core::Error::Msg("Invalid data URL format".to_string()))?
    } else {
        b64
    };

    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| candle_core::Error::Msg(format!("Base64 decode error: {}", e)))
}

/// Decode audio from a MediaSourceStream using symphonia.
///
/// This is the shared implementation for all audio loading functions.
#[cfg(feature = "audio-loading")]
fn decode_audio_stream(
    mss: symphonia::core::io::MediaSourceStream,
    hint: symphonia::core::probe::Hint,
) -> Result<AudioData> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::meta::MetadataOptions;

    // Probe the media source
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| candle_core::Error::Msg(format!("Failed to probe audio format: {}", e)))?;

    let mut format = probed.format;

    // Get the default track
    let track = format
        .default_track()
        .ok_or_else(|| candle_core::Error::Msg("No audio tracks found".to_string()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| candle_core::Error::Msg("Unknown sample rate".to_string()))?;

    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    // Create a decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create decoder: {}", e)))?;

    let track_id = track.id;
    let mut samples = Vec::new();

    // Decode all packets
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(candle_core::Error::Msg(format!("Decode error: {}", e))),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(candle_core::Error::Msg(format!("Decode error: {}", e))),
        };

        // Convert to f32 samples
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
        sample_buf.copy_interleaved_ref(decoded);
        samples.extend(sample_buf.samples());
    }

    Ok(AudioData::new(samples, sample_rate, channels))
}

/// Load audio from a file path.
///
/// Supports wav, mp3, flac, ogg, and other formats via symphonia.
#[cfg(feature = "audio-loading")]
pub fn load_audio_file(path: &str) -> Result<AudioData> {
    use std::fs::File;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let file = File::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open {}: {}", path, e)))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create a hint from the file extension
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension() {
        hint.with_extension(ext.to_str().unwrap_or(""));
    }

    decode_audio_stream(mss, hint)
}

/// Load audio from base64 string.
#[cfg(feature = "audio-loading")]
pub fn load_audio_base64(b64: &str) -> Result<AudioData> {
    use std::io::Cursor;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let bytes = decode_base64_to_bytes(b64)?;
    let cursor = Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    decode_audio_stream(mss, Hint::new())
}

/// Load audio from a URL.
///
/// Fetches the audio file from the URL and decodes it.
/// Supports the same formats as `load_audio_file` (wav, mp3, flac, ogg, etc.).
#[cfg(feature = "audio-loading")]
pub fn load_audio_url(url: &str) -> Result<AudioData> {
    use std::io::Cursor;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    // Fetch the audio data from URL
    let response = reqwest::blocking::get(url)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to fetch URL {}: {}", url, e)))?;

    if !response.status().is_success() {
        return Err(candle_core::Error::Msg(format!(
            "HTTP error fetching {}: {}",
            url,
            response.status()
        )));
    }

    let bytes = response
        .bytes()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read response body: {}", e)))?;

    // Create a hint from the URL extension if available
    let mut hint = Hint::new();
    if let Some(ext) = url.rsplit('.').next() {
        // Only use extension if it looks like an audio format
        let ext_lower = ext.to_lowercase();
        if ["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus", "webm"].contains(&ext_lower.as_str())
        {
            hint.with_extension(&ext_lower);
        }
    }

    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    decode_audio_stream(mss, hint)
}

/// Stub for URL loading when audio-loading feature is disabled.
#[cfg(not(feature = "audio-loading"))]
pub fn load_audio_url(_url: &str) -> Result<AudioData> {
    Err(candle_core::Error::Msg(
        "URL audio loading requires the 'audio-loading' feature".to_string(),
    ))
}

/// Resample audio to a target sample rate.
///
/// Uses the rubato library for high-quality resampling.
#[cfg(feature = "audio-loading")]
pub fn resample(audio: &AudioData, target_sample_rate: u32) -> Result<AudioData> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if audio.sample_rate == target_sample_rate {
        return Ok(audio.clone());
    }

    // Ensure mono for simplicity (TTS typically uses mono)
    let mono = audio.to_mono();

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let resample_ratio = target_sample_rate as f64 / mono.sample_rate as f64;

    let mut resampler = SincFixedIn::<f32>::new(
        resample_ratio,
        2.0, // max relative ratio deviation
        params,
        mono.samples.len(),
        1, // channels
    )
    .map_err(|e| candle_core::Error::Msg(format!("Resampler creation failed: {}", e)))?;

    let waves_in = vec![mono.samples.clone()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| candle_core::Error::Msg(format!("Resampling failed: {}", e)))?;

    Ok(AudioData::new(
        waves_out.into_iter().next().unwrap_or_default(),
        target_sample_rate,
        1,
    ))
}

/// Load audio from any supported source.
///
/// Automatically detects the input type:
/// - File path: loads from disk
/// - Base64 string: decodes and loads
/// - URL: fetches and loads from HTTP/HTTPS
///
/// Returns mono audio resampled to the target sample rate if specified.
#[cfg(feature = "audio-loading")]
pub fn load_audio(input: AudioInput, target_sample_rate: Option<u32>) -> Result<AudioData> {
    let audio = match input {
        AudioInput::FilePath(path) => {
            if is_url(&path) {
                load_audio_url(&path)?
            } else {
                load_audio_file(&path)?
            }
        }
        AudioInput::Base64(b64) => load_audio_base64(&b64)?,
        AudioInput::Raw {
            samples,
            sample_rate,
            channels,
        } => AudioData::new(samples, sample_rate, channels),
        AudioInput::AudioData(data) => data,
    };

    // Convert to mono
    let mono = audio.to_mono();

    // Resample if target rate specified
    match target_sample_rate {
        Some(rate) => resample(&mono, rate),
        None => Ok(mono),
    }
}

/// Load audio and convert to tensor.
///
/// Returns a 1D tensor of shape `(samples,)` with f32 values in [-1, 1].
#[cfg(feature = "audio-loading")]
pub fn load_audio_tensor(
    input: AudioInput,
    target_sample_rate: Option<u32>,
    device: &Device,
) -> Result<(Tensor, u32)> {
    let audio = load_audio(input, target_sample_rate)?;
    let tensor = audio.to_tensor(device)?;
    Ok((tensor, audio.sample_rate))
}

// Stub implementations when audio-loading feature is not enabled
#[cfg(not(feature = "audio-loading"))]
pub fn load_audio_file(_path: &str) -> Result<AudioData> {
    Err(candle_core::Error::Msg(
        "Audio loading requires the 'audio-loading' feature".to_string(),
    ))
}

#[cfg(not(feature = "audio-loading"))]
pub fn load_audio_base64(_b64: &str) -> Result<AudioData> {
    Err(candle_core::Error::Msg(
        "Audio loading requires the 'audio-loading' feature".to_string(),
    ))
}

#[cfg(not(feature = "audio-loading"))]
pub fn resample(_audio: &AudioData, _target_sample_rate: u32) -> Result<AudioData> {
    Err(candle_core::Error::Msg(
        "Audio resampling requires the 'audio-loading' feature".to_string(),
    ))
}

#[cfg(not(feature = "audio-loading"))]
pub fn load_audio(_input: AudioInput, _target_sample_rate: Option<u32>) -> Result<AudioData> {
    Err(candle_core::Error::Msg(
        "Audio loading requires the 'audio-loading' feature".to_string(),
    ))
}

#[cfg(not(feature = "audio-loading"))]
pub fn load_audio_tensor(
    _input: AudioInput,
    _target_sample_rate: Option<u32>,
    _device: &Device,
) -> Result<(Tensor, u32)> {
    Err(candle_core::Error::Msg(
        "Audio loading requires the 'audio-loading' feature".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_probably_base64() {
        // Not base64 - file path
        assert!(!is_probably_base64("/path/to/audio.wav"));
        assert!(!is_probably_base64("audio.wav"));

        // Data URL format
        assert!(is_probably_base64("data:audio/wav;base64,UklGRi..."));

        // Short string - not base64
        assert!(!is_probably_base64("SGVsbG8gV29ybGQ="));
    }

    #[test]
    fn test_is_url() {
        assert!(is_url("http://example.com/audio.wav"));
        assert!(is_url("https://example.com/audio.wav"));
        assert!(!is_url("/path/to/audio.wav"));
        assert!(!is_url("audio.wav"));
    }

    #[test]
    fn test_audio_data_to_mono() {
        // Stereo to mono
        let stereo = AudioData::new(vec![0.5, -0.5, 0.8, -0.8, 0.3, -0.3], 44100, 2);
        let mono = stereo.to_mono();
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples.len(), 3);
        assert!((mono.samples[0] - 0.0).abs() < 1e-6); // (0.5 + -0.5) / 2
        assert!((mono.samples[1] - 0.0).abs() < 1e-6);
        assert!((mono.samples[2] - 0.0).abs() < 1e-6);
    }

    #[cfg(feature = "audio-loading")]
    #[test]
    fn test_resample_changes_length() {
        let audio = AudioData::new(vec![0.1f32; 48000], 48000, 1);
        let resampled = super::resample(&audio, 24000).expect("resample should succeed");
        assert_eq!(resampled.sample_rate, 24000);
        let diff = (resampled.samples.len() as i64 - 24000i64).abs();
        assert!(
            diff <= 100,
            "expected ~24000 samples, got {} (diff {})",
            resampled.samples.len(),
            diff
        );
    }

    #[cfg(feature = "audio-loading")]
    #[test]
    fn test_resample_noop_same_rate() {
        let audio = AudioData::new(vec![0.5f32; 24000], 24000, 1);
        let resampled = super::resample(&audio, 24000).expect("resample should succeed");
        assert_eq!(resampled.samples.len(), 24000);
        assert_eq!(resampled.sample_rate, 24000);
    }

    #[test]
    fn test_audio_data_duration() {
        let audio = AudioData::new(vec![0.0; 48000], 48000, 1);
        assert!((audio.duration_secs() - 1.0).abs() < 1e-6);

        let stereo = AudioData::new(vec![0.0; 96000], 48000, 2);
        assert!((stereo.duration_secs() - 1.0).abs() < 1e-6);
    }
}
