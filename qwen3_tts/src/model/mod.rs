//! High-level TTS model wrapper.
//!
//! This module provides a user-friendly interface for Qwen3-TTS, supporting:
//! - Voice cloning from reference audio (x-vector and ICL modes)
//! - Custom voice from predefined speakers
//! - Voice design from text descriptions
//!
//! # Example
//!
//! ```no_run
//! use qwen3_tts::model::loader::{ModelLoader, LoaderConfig};
//! use candle_core::Device;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let loader = ModelLoader::from_local_dir("/path/to/model")?;
//! let model = loader.load_tts_model(&Device::Cpu, &LoaderConfig::default())?;
//!
//! // Generate speech with custom voice
//! let result = model.generate_custom_voice_from_text(
//!     "Hello, world!",
//!     "vivian",
//!     "english",
//!     None,
//!     None,
//! )?;
//! let audio = result.audio;
//! let sample_rate = result.sample_rate;
//! # Ok(())
//! # }

pub mod config;
pub mod custom_voice;
pub mod generate;
pub mod loader;
pub mod options;
pub mod types;
pub mod voice_clone;
pub mod voice_design;

use crate::{
    audio::{
        mel::{MelSpectrogramConfig, mel_spectrogram},
        tokenizer::v2::TokenizerV2,
    },
    model::{
        config::GenerateConfig,
        types::{SUPPORTED_LANGUAGES, TTSModelType},
        voice_clone::VoiceClonePromptItem,
    },
    nn::generation::ConditionalGeneration,
    text::processing::{PaddingSide, TextProcessor, TokenizerOutput},
};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use std::collections::HashMap;

// =========================================================================
// Validation Helper Functions (testable without model loading)
// =========================================================================

/// Validate that a language is in the supported languages list.
///
/// # Arguments
/// * `language` - Language code to validate
///
/// # Returns
/// `Ok(())` if valid, `Err` with message if not.
pub fn validate_language_value(language: &str) -> Result<()> {
    if SUPPORTED_LANGUAGES.contains(&language) {
        Ok(())
    } else {
        Err(candle_core::Error::Msg(format!(
            "Unsupported language '{}'. Supported: {:?}",
            language, SUPPORTED_LANGUAGES
        )))
    }
}

/// Validate that a speaker is defined in the given speaker ID map.
///
/// # Arguments
/// * `speaker` - Speaker name to validate
/// * `spk_id` - Optional map of speaker names to IDs
///
/// # Returns
/// `Ok(())` if valid, `Err` with message if not.
pub fn validate_speaker_value(
    speaker: &str,
    spk_id: &Option<HashMap<String, usize>>,
) -> Result<()> {
    match spk_id {
        Some(spk_map) if spk_map.contains_key(speaker) => Ok(()),
        Some(spk_map) => {
            let available: Vec<_> = spk_map.keys().collect();
            Err(candle_core::Error::Msg(format!(
                "Unknown speaker '{}'. Available: {:?}",
                speaker, available
            )))
        }
        None => Err(candle_core::Error::Msg(
            "No speakers defined in model config (spk_id is None)".to_string(),
        )),
    }
}

/// High-level wrapper for Qwen3-TTS inference.
///
/// Provides a simple interface for:
/// - Voice cloning from reference audio (Base model)
/// - Custom voice from predefined speakers (CustomVoice model)
/// - Voice design from text descriptions (VoiceDesign model)
///
/// # Model Types
///
/// The wrapper automatically detects the model type from config and validates
/// that the correct generation API is used:
/// - `generate_voice_clone()` - Only for Base models
/// - `generate_custom_voice()` - Only for CustomVoice models
/// - `generate_voice_design()` - Only for VoiceDesign models
///
/// # Voice Cloning Modes (Base model only)
///
/// - **X-vector only mode**: Uses only the speaker embedding extracted from
///   reference audio. Simpler but may not capture prosodic patterns as well.
///
/// - **ICL (In-Context Learning) mode**: Uses both the speaker embedding and
///   learns from the reference audio/transcript pair. Typically produces better
///   results for matching prosody and speaking style.
pub struct Model {
    model: ConditionalGeneration,
    audio_tokenizer: Option<TokenizerV2>,
    text_processor: Option<TextProcessor>,
    mel_config: MelSpectrogramConfig,
    device: Device,
    /// Data type for model tensors (used for dtype conversion).
    dtype: DType,
    /// Default generation parameters (from generate_config.json or defaults).
    generate_defaults: GenerateConfig,
    /// Model type for API validation.
    model_type: TTSModelType,
}

/// Configuration for the speaker encoder.
const SPEAKER_ENCODER_SAMPLE_RATE: u32 = 24000;

impl Model {
    /// Create a new TTS model.
    pub fn new(
        model: ConditionalGeneration,
        audio_tokenizer: Option<TokenizerV2>,
        device: Device,
        dtype: DType,
    ) -> Self {
        let model_type = model
            .get_config()
            .tts_model_type
            .as_ref()
            .map(|s| TTSModelType::parse(s))
            .unwrap_or(TTSModelType::Unknown);

        Self {
            model,
            audio_tokenizer,
            text_processor: None,
            mel_config: MelSpectrogramConfig::default(),
            device,
            dtype,
            generate_defaults: GenerateConfig::default(),
            model_type,
        }
    }

    /// Create a new TTS model with custom mel spectrogram config.
    pub fn with_mel_config(
        model: ConditionalGeneration,
        audio_tokenizer: Option<TokenizerV2>,
        mel_config: MelSpectrogramConfig,
        device: Device,
        dtype: DType,
    ) -> Self {
        let model_type = model
            .get_config()
            .tts_model_type
            .as_ref()
            .map(|s| TTSModelType::parse(s))
            .unwrap_or(TTSModelType::Unknown);

        Self {
            model,
            audio_tokenizer,
            text_processor: None,
            mel_config,
            device,
            dtype,
            generate_defaults: GenerateConfig::default(),
            model_type,
        }
    }

    /// Create a new TTS model with generation config defaults.
    pub fn with_generate_config(
        model: ConditionalGeneration,
        audio_tokenizer: Option<TokenizerV2>,
        generate_defaults: GenerateConfig,
        device: Device,
        dtype: DType,
    ) -> Self {
        let model_type = model
            .get_config()
            .tts_model_type
            .as_ref()
            .map(|s| TTSModelType::parse(s))
            .unwrap_or(TTSModelType::Unknown);

        Self {
            model,
            audio_tokenizer,
            text_processor: None,
            mel_config: MelSpectrogramConfig::default(),
            device,
            dtype,
            generate_defaults,
            model_type,
        }
    }

    /// Create a new TTS model with all components.
    pub fn with_all(
        model: ConditionalGeneration,
        audio_tokenizer: Option<TokenizerV2>,
        text_processor: Option<TextProcessor>,
        generate_defaults: GenerateConfig,
        device: Device,
        dtype: DType,
    ) -> Self {
        let model_type = model
            .get_config()
            .tts_model_type
            .as_ref()
            .map(|s| TTSModelType::parse(s))
            .unwrap_or(TTSModelType::Unknown);

        Self {
            model,
            audio_tokenizer,
            text_processor,
            mel_config: MelSpectrogramConfig::default(),
            device,
            dtype,
            generate_defaults,
            model_type,
        }
    }

    /// Create a new TTS model loading generate config from a model directory.
    ///
    /// Automatically loads `generate_config.json` from the model directory if present.
    /// Falls back to defaults if not found.
    pub fn from_model_dir(
        model: ConditionalGeneration,
        audio_tokenizer: Option<TokenizerV2>,
        model_dir: impl AsRef<std::path::Path>,
        device: Device,
        dtype: DType,
    ) -> Self {
        let generate_defaults = GenerateConfig::from_model_dir(&model_dir);

        // Try to load text tokenizer from the model directory
        let text_processor = Self::try_load_text_processor(&model_dir);

        Self::with_all(
            model,
            audio_tokenizer,
            text_processor,
            generate_defaults,
            device,
            dtype,
        )
    }

    /// Try to load a text processor from a model directory.
    /// Try to load a text processor from a model directory.
    ///
    /// Supports multiple formats:
    /// - `tokenizer.json` (HuggingFace fast tokenizer)
    /// - `vocab.json` + `merges.txt` (BPE format, used by Qwen models)
    fn try_load_text_processor(model_dir: impl AsRef<std::path::Path>) -> Option<TextProcessor> {
        let model_dir = model_dir.as_ref();

        match TextProcessor::from_pretrained(model_dir) {
            Ok(processor) => {
                tracing::info!("Loaded text tokenizer from {}", model_dir.display());
                Some(processor)
            }
            Err(e) => {
                tracing::debug!("No text tokenizer found in {}: {}", model_dir.display(), e);
                None
            }
        }
    }

    /// Get the underlying model.
    pub fn model(&self) -> &ConditionalGeneration {
        &self.model
    }

    /// Get the audio tokenizer.
    pub fn audio_tokenizer(&self) -> Option<&TokenizerV2> {
        self.audio_tokenizer.as_ref()
    }

    /// Get the text processor.
    pub fn text_processor(&self) -> Option<&TextProcessor> {
        self.text_processor.as_ref()
    }

    /// Check if the model has a text processor loaded.
    pub fn has_text_processor(&self) -> bool {
        self.text_processor.is_some()
    }

    /// Set the text processor.
    ///
    /// Useful when loading a text tokenizer separately or replacing the existing one.
    pub fn set_text_processor(&mut self, processor: TextProcessor) {
        self.text_processor = Some(processor);
    }

    // =========================================================================
    // Text Tokenization Methods
    // =========================================================================

    /// Tokenize a single text for TTS generation.
    ///
    /// The text is wrapped in the assistant chat template before tokenizing.
    ///
    /// # Arguments
    /// * `text` - The text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs, or error if no text processor is loaded.
    pub fn tokenize_text(&self, text: &str) -> std::result::Result<Vec<u32>, String> {
        match &self.text_processor {
            Some(processor) => Ok(processor.tokenize_for_tts(text)),
            None => Err("No text processor loaded. Load a tokenizer.json file first.".to_string()),
        }
    }

    /// Tokenize multiple texts for batched TTS generation with automatic padding.
    ///
    /// The texts are wrapped in chat templates and padded to the same length.
    /// Uses left-padding by default (required for autoregressive generation).
    ///
    /// # Arguments
    /// * `texts` - The texts to tokenize
    ///
    /// # Returns
    /// `TokenizerOutput` with padded input_ids, attention_mask, and original lengths.
    pub fn tokenize_texts(&self, texts: &[&str]) -> std::result::Result<TokenizerOutput, String> {
        match &self.text_processor {
            Some(processor) => processor
                .batch_tokenize_for_tts(texts, PaddingSide::Left)
                .map_err(|e| e.to_string()),
            None => Err("No text processor loaded. Load a tokenizer.json file first.".to_string()),
        }
    }

    /// Tokenize texts with custom padding side.
    ///
    /// # Arguments
    /// * `texts` - The texts to tokenize
    /// * `padding_side` - Whether to pad on left or right
    ///
    /// # Returns
    /// `TokenizerOutput` with padded sequences.
    pub fn tokenize_texts_with_padding(
        &self,
        texts: &[&str],
        padding_side: PaddingSide,
    ) -> std::result::Result<TokenizerOutput, String> {
        match &self.text_processor {
            Some(processor) => processor
                .batch_tokenize_for_tts(texts, padding_side)
                .map_err(|e| e.to_string()),
            None => Err("No text processor loaded. Load a tokenizer.json file first.".to_string()),
        }
    }

    /// Convert tokenized output to a Tensor suitable for model input.
    ///
    /// # Arguments
    /// * `output` - The tokenizer output
    ///
    /// # Returns
    /// A 2D tensor of shape (batch_size, seq_len) with token IDs.
    pub fn tokens_to_tensor(&self, output: &TokenizerOutput) -> Result<Tensor> {
        let batch_size = output.input_ids.len();
        if batch_size == 0 {
            return Tensor::zeros((0, 0), DType::U32, &self.device);
        }
        let seq_len = output.input_ids[0].len();
        let flat: Vec<u32> = output.input_ids.iter().flatten().copied().collect();
        Tensor::from_vec(flat, (batch_size, seq_len), &self.device)
    }

    /// Convert attention mask to a Tensor.
    ///
    /// # Arguments
    /// * `output` - The tokenizer output
    ///
    /// # Returns
    /// A 2D tensor of shape (batch_size, seq_len) with the attention mask.
    pub fn attention_mask_to_tensor(&self, output: &TokenizerOutput) -> Result<Tensor> {
        let batch_size = output.attention_mask.len();
        if batch_size == 0 {
            return Tensor::zeros((0, 0), DType::U32, &self.device);
        }
        let seq_len = output.attention_mask[0].len();
        let flat: Vec<u32> = output.attention_mask.iter().flatten().copied().collect();
        Tensor::from_vec(flat, (batch_size, seq_len), &self.device)
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model's data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the mel spectrogram configuration.
    pub fn mel_config(&self) -> &MelSpectrogramConfig {
        &self.mel_config
    }

    /// Get the sample rate expected for speaker encoder input.
    pub fn speaker_encoder_sample_rate(&self) -> u32 {
        SPEAKER_ENCODER_SAMPLE_RATE
    }

    /// Get the codec tokenizer rate in Hz (e.g. 12 for 12Hz, 25 for 25Hz).
    pub fn tokenizer_rate(&self) -> usize {
        self.model.get_config().tokenizer_rate()
    }

    /// Get the model type.
    pub fn model_type(&self) -> TTSModelType {
        self.model_type
    }

    // ===== Model Type Validation =====

    /// Check if this is a Base model (supports voice cloning).
    pub fn is_base_model(&self) -> bool {
        matches!(self.model_type, TTSModelType::Base | TTSModelType::Unknown)
    }

    /// Check if this is a CustomVoice model (supports predefined speakers).
    pub fn is_custom_voice_model(&self) -> bool {
        matches!(
            self.model_type,
            TTSModelType::CustomVoice | TTSModelType::Unknown
        )
    }

    /// Check if this is a VoiceDesign model (supports text descriptions).
    pub fn is_voice_design_model(&self) -> bool {
        matches!(
            self.model_type,
            TTSModelType::VoiceDesign | TTSModelType::Unknown
        )
    }

    /// Validate that the model supports voice cloning.
    fn require_base_model(&self) -> Result<()> {
        if !self.is_base_model() {
            return Err(candle_core::Error::Msg(format!(
                "Model type {:?} does not support generate_voice_clone(). \
                 Only Base models support voice cloning.",
                self.model_type
            )));
        }
        Ok(())
    }

    /// Validate that the model supports custom voice generation.
    fn require_custom_voice_model(&self) -> Result<()> {
        if !self.is_custom_voice_model() {
            return Err(candle_core::Error::Msg(format!(
                "Model type {:?} does not support generate_custom_voice(). \
                 Only CustomVoice models support predefined speakers.",
                self.model_type
            )));
        }
        Ok(())
    }

    /// Validate that the model supports voice design.
    fn require_voice_design_model(&self) -> Result<()> {
        if !self.is_voice_design_model() {
            return Err(candle_core::Error::Msg(format!(
                "Model type {:?} does not support generate_voice_design(). \
                 Only VoiceDesign models support text-based voice descriptions.",
                self.model_type
            )));
        }
        Ok(())
    }

    // ===== Voice Clone Prompt Creation =====

    /// Compute mel spectrogram from audio waveform.
    ///
    /// The audio should be at the sample rate specified in `mel_config`
    /// (default 24kHz for speaker encoder).
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio waveform tensor of shape `(samples,)` or `(batch, samples)`
    ///
    /// # Returns
    ///
    /// Mel spectrogram of shape `(batch, time, num_mels)`
    pub fn compute_mel_spectrogram(&self, audio: &Tensor) -> Result<Tensor> {
        // mel_spectrogram returns (batch, num_mels, time), we need (batch, time, num_mels)
        // Note: mel_spectrogram expects F32 audio and returns F32
        let mel = mel_spectrogram(audio, &self.mel_config, &self.device)?;
        // Transpose to (batch, time, num_mels) for speaker encoder
        let mel = mel.permute((0, 2, 1))?;
        // Convert to model dtype for speaker encoder compatibility
        mel.to_dtype(self.dtype)
    }

    /// Extract speaker embedding from audio waveform.
    ///
    /// This is a convenience method that:
    /// 1. Computes the mel spectrogram
    /// 2. Extracts the speaker embedding
    ///
    /// The audio should be at 24kHz (speaker encoder sample rate).
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio waveform tensor of shape `(samples,)` or `(batch, samples)`
    ///
    /// # Returns
    ///
    /// Speaker embedding of shape `(batch, enc_dim)` or `(enc_dim,)` for unbatched input
    pub fn extract_speaker_embedding(&self, audio: &Tensor) -> Result<Tensor> {
        let was_1d = audio.dims().len() == 1;
        let mel = self.compute_mel_spectrogram(audio)?;
        let embedding = self.model.encode_speaker(&mel)?;

        if was_1d {
            // Remove batch dimension for consistency with input
            embedding.squeeze(0)
        } else {
            Ok(embedding)
        }
    }

    /// Create a voice clone prompt from a mel spectrogram.
    ///
    /// This method extracts the speaker embedding from the mel spectrogram
    /// and creates a `VoiceClonePromptItem` for use in generation.
    ///
    /// # Arguments
    ///
    /// * `mel` - Mel spectrogram of shape `(time, num_mels)` or `(batch, time, num_mels)`
    /// * `ref_text` - Optional reference transcript (required for ICL mode)
    /// * `x_vector_only_mode` - If true, only use speaker embedding (no ICL)
    ///
    /// # Returns
    ///
    /// A `VoiceClonePromptItem` ready for generation.
    ///
    /// # Note
    ///
    /// ICL mode requires audio codes (`ref_code`) which are obtained by encoding
    /// the reference audio through the audio tokenizer. The audio tokenizer encoder
    /// is not yet implemented, so ICL mode currently only creates the speaker
    /// embedding portion of the prompt.
    pub fn create_voice_clone_prompt_from_mel(
        &self,
        mel: &Tensor,
        ref_text: Option<String>,
        x_vector_only_mode: bool,
    ) -> Result<VoiceClonePromptItem> {
        // Ensure mel is 3D: (batch, time, num_mels)
        let mel = if mel.dims().len() == 2 {
            mel.unsqueeze(0)?
        } else {
            mel.clone()
        };

        // Extract speaker embedding
        let speaker_embed = self.model.encode_speaker(&mel)?;
        // Remove batch dimension since we typically process one reference at a time
        let speaker_embed = speaker_embed.squeeze(0)?;

        if x_vector_only_mode {
            Ok(VoiceClonePromptItem::x_vector_only(speaker_embed))
        } else {
            // This method only has the mel spectrogram, not the raw audio,
            // so it cannot encode reference audio codes for ICL mode.
            // Use `create_voice_clone_prompt_from_audio` for full ICL support.
            Ok(VoiceClonePromptItem::new(
                None,
                speaker_embed,
                x_vector_only_mode,
                ref_text,
            ))
        }
    }

    /// Create a voice clone prompt from audio waveform.
    ///
    /// This is the recommended method for creating voice clone prompts as it:
    /// 1. Computes the mel spectrogram for speaker embedding
    /// 2. Encodes the audio to codes for ICL mode (if encoder available)
    /// 3. Creates a complete `VoiceClonePromptItem`
    ///
    /// The audio should be at 24kHz (speaker encoder sample rate).
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio waveform tensor of shape `(samples,)`
    /// * `ref_text` - Optional reference transcript (required for ICL mode)
    /// * `x_vector_only_mode` - If true, only use speaker embedding (no ICL)
    ///
    /// # Returns
    ///
    /// A `VoiceClonePromptItem` ready for generation.
    ///
    /// # ICL Mode
    ///
    /// When `x_vector_only_mode` is false, this method will:
    /// - Encode the audio to discrete codes using the audio tokenizer (if available)
    /// - Store the codes in `ref_code` for in-context learning
    /// - The model will learn from both the speaker embedding and the audio patterns
    ///
    /// If the audio tokenizer encoder is not available, ICL mode will fall back
    /// to x-vector only mode with a warning.
    pub fn create_voice_clone_prompt_from_audio(
        &mut self,
        audio: &Tensor,
        ref_text: Option<String>,
        x_vector_only_mode: bool,
    ) -> Result<VoiceClonePromptItem> {
        // Compute mel spectrogram for speaker embedding
        let mel = self.compute_mel_spectrogram(audio)?;

        // Debug: print mel spectrogram stats
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(shape = ?mel.dims(), dtype = ?mel.dtype(), "Mel spectrogram");
        }
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(mel_f32) = mel.to_dtype(DType::F32)
            && let Ok(mel_flat) = mel_f32.flatten_all()
            && let Ok(values) = mel_flat.to_vec1::<f32>()
        {
            let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = values.iter().sum();
            let mean = sum / values.len() as f32;
            tracing::debug!(
                min = format!("{:.4}", min),
                max = format!("{:.4}", max),
                mean_val = format!("{:.4}", mean),
                "Mel stats"
            );
        }
        // Print first 5 mel values at time=0 for comparison with Python
        if tracing::enabled!(tracing::Level::DEBUG)
            && let Ok(mel_f32) = mel.to_dtype(DType::F32)
            && let Ok(first_frame) = mel_f32.i((0, 0, ..5))
            && let Ok(first5) = first_frame.to_vec1::<f32>()
        {
            tracing::debug!(?first5, "Mel first 5 values (time=0)");
        }

        // Extract speaker embedding
        let speaker_embed = self.model.encode_speaker(&mel)?;
        let speaker_embed = speaker_embed.squeeze(0)?;

        if x_vector_only_mode {
            return Ok(VoiceClonePromptItem::x_vector_only(speaker_embed));
        }

        // ICL mode: try to encode audio to codes
        let ref_code = if let Some(ref mut tokenizer) = self.audio_tokenizer {
            if tokenizer.has_encoder() {
                // Ensure audio is 2D: (batch, samples) and convert to model dtype
                let audio_batched = if audio.dims().len() == 1 {
                    audio.unsqueeze(0)?.to_dtype(self.dtype)?
                } else {
                    audio.to_dtype(self.dtype)?
                };

                // Encode audio to discrete codes
                // The tokenizer's encode returns a Tensor of shape (batch, num_quantizers, seq_len)
                match tokenizer.encode(&audio_batched) {
                    Ok(codes) => {
                        // Transform shape: (batch, num_quantizers, seq_len) -> (seq_len, num_quantizers)
                        // 1. Squeeze batch dimension: (num_quantizers, seq_len)
                        // 2. Transpose to match expected format: (seq_len, num_quantizers)
                        match codes.squeeze(0).and_then(|c| c.transpose(0, 1)) {
                            Ok(transposed) => {
                                tracing::debug!(
                                    shape = ?transposed.dims(),
                                    "Encoded reference audio to codes"
                                );
                                Some(transposed)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "Failed to reshape encoded codes. Falling back to x-vector only mode."
                                );
                                None
                            }
                        }
                    }
                    Err(e) => {
                        // Log warning and continue without codes
                        tracing::warn!(
                            error = %e,
                            "Failed to encode audio for ICL mode. Falling back to x-vector only mode."
                        );
                        None
                    }
                }
            } else {
                // Encoder not available
                tracing::warn!(
                    "Audio tokenizer encoder not available for ICL mode. Falling back to x-vector only mode."
                );
                None
            }
        } else {
            // No tokenizer at all
            tracing::warn!(
                "No audio tokenizer available for ICL mode. Falling back to x-vector only mode."
            );
            None
        };

        // Create prompt item
        // ICL mode requires BOTH ref_code AND ref_text
        let icl_mode = ref_code.is_some() && ref_text.is_some();

        // Warn if we have audio codes but no ref_text (fallback to x-vector only)
        if ref_code.is_some() && ref_text.is_none() {
            tracing::warn!(
                "Reference audio was encoded but no --ref-text provided. \
                 Falling back to x-vector only mode. For higher quality voice cloning, \
                 provide --ref-text with the transcript of the reference audio."
            );
        }

        Ok(VoiceClonePromptItem {
            ref_code: if icl_mode { ref_code } else { None },
            ref_spk_embedding: speaker_embed,
            x_vector_only_mode: !icl_mode,
            icl_mode,
            ref_text,
        })
    }

    /// Create a voice clone prompt from audio waveform with explicit sample rate.
    ///
    /// This method handles resampling if the audio is not at 24kHz.
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio waveform tensor of shape `(samples,)`
    /// * `sample_rate` - Sample rate of the input audio
    /// * `ref_text` - Optional reference transcript (required for ICL mode)
    /// * `x_vector_only_mode` - If true, only use speaker embedding (no ICL)
    ///
    /// # Returns
    ///
    /// A `VoiceClonePromptItem` ready for generation.
    ///
    /// # Note
    ///
    /// If the sample rate is not 24kHz, the audio is automatically resampled.
    /// Resampling requires the `audio-loading` feature (for rubato).
    pub fn create_voice_clone_prompt_with_sample_rate(
        &mut self,
        audio: &Tensor,
        sample_rate: u32,
        ref_text: Option<String>,
        x_vector_only_mode: bool,
    ) -> Result<VoiceClonePromptItem> {
        let audio = if sample_rate != 24000 {
            let samples = audio.to_vec1::<f32>()?;
            let audio_data = crate::audio::utils::AudioData {
                samples,
                sample_rate,
                channels: 1,
            };
            let resampled = crate::audio::utils::resample(&audio_data, 24000)?;
            let len = resampled.samples.len();
            Tensor::from_vec(resampled.samples, len, audio.device())?
        } else {
            audio.clone()
        };
        self.create_voice_clone_prompt_from_audio(&audio, ref_text, x_vector_only_mode)
    }

    // ===== Validation =====

    /// Validate that the language is supported.
    ///
    /// # Arguments
    ///
    /// * `language` - Language code to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with message if not.
    pub fn validate_language(&self, language: &str) -> Result<()> {
        validate_language_value(language)
    }

    /// Validate that the speaker is defined in the model config.
    ///
    /// # Arguments
    ///
    /// * `speaker` - Speaker name to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with message if not.
    pub fn validate_speaker(&self, speaker: &str) -> Result<()> {
        let config = self.model.get_config();
        validate_speaker_value(speaker, &config.talker_config.spk_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Language Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_language_all_supported() {
        // All 8 supported languages should pass validation
        for lang in SUPPORTED_LANGUAGES {
            let result = validate_language_value(lang);
            assert!(
                result.is_ok(),
                "Language '{}' should be valid but got: {:?}",
                lang,
                result
            );
        }
    }

    #[test]
    fn test_validate_language_invalid() {
        let result = validate_language_value("esperanto");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unsupported language"),
            "Error should mention unsupported language"
        );
        assert!(
            err_msg.contains("esperanto"),
            "Error should contain the invalid language"
        );
        // Should list supported languages
        assert!(
            err_msg.contains("english"),
            "Error should list supported languages"
        );
    }

    #[test]
    fn test_validate_language_case_sensitive() {
        // Language validation is case-sensitive - "English" != "english"
        let result = validate_language_value("English");
        assert!(
            result.is_err(),
            "Language validation should be case-sensitive"
        );

        let result = validate_language_value("CHINESE");
        assert!(
            result.is_err(),
            "Language validation should be case-sensitive"
        );
    }

    #[test]
    fn test_validate_language_empty_string() {
        let result = validate_language_value("");
        assert!(
            result.is_err(),
            "Empty string should not be a valid language"
        );
    }

    // =========================================================================
    // Speaker Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_speaker_valid() {
        let mut spk_map = HashMap::new();
        spk_map.insert("alice".to_string(), 0);
        spk_map.insert("bob".to_string(), 1);
        let spk_id = Some(spk_map);

        let result = validate_speaker_value("alice", &spk_id);
        assert!(result.is_ok(), "Valid speaker should pass validation");

        let result = validate_speaker_value("bob", &spk_id);
        assert!(result.is_ok(), "Valid speaker should pass validation");
    }

    #[test]
    fn test_validate_speaker_invalid() {
        let mut spk_map = HashMap::new();
        spk_map.insert("alice".to_string(), 0);
        spk_map.insert("bob".to_string(), 1);
        let spk_id = Some(spk_map);

        let result = validate_speaker_value("charlie", &spk_id);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unknown speaker"),
            "Error should mention unknown speaker"
        );
        assert!(
            err_msg.contains("charlie"),
            "Error should contain the invalid speaker"
        );
        // Should list available speakers (order may vary due to HashMap)
        assert!(
            err_msg.contains("alice") || err_msg.contains("bob"),
            "Error should list available speakers"
        );
    }

    #[test]
    fn test_validate_speaker_none_spk_id() {
        let spk_id: Option<HashMap<String, usize>> = None;

        let result = validate_speaker_value("anyone", &spk_id);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No speakers defined"),
            "Error should indicate no speakers are defined"
        );
    }

    #[test]
    fn test_validate_speaker_empty_map() {
        let spk_map: HashMap<String, usize> = HashMap::new();
        let spk_id = Some(spk_map);

        let result = validate_speaker_value("anyone", &spk_id);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unknown speaker"),
            "Error should mention unknown speaker"
        );
    }

    #[test]
    fn test_validate_speaker_case_sensitive() {
        let mut spk_map = HashMap::new();
        spk_map.insert("alice".to_string(), 0);
        let spk_id = Some(spk_map);

        // Speaker validation is case-sensitive
        let result = validate_speaker_value("Alice", &spk_id);
        assert!(
            result.is_err(),
            "Speaker validation should be case-sensitive"
        );

        let result = validate_speaker_value("ALICE", &spk_id);
        assert!(
            result.is_err(),
            "Speaker validation should be case-sensitive"
        );
    }
}
