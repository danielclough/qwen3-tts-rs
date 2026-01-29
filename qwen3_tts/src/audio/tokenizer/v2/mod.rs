//! Modeling components for the 12Hz tokenizer.

pub mod causal_conv;
pub mod config;
pub mod convnext;
pub mod transformer;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::audio::{
    decoder::v2::TokenizerV2Decoder,
    encoder::v2::{TokenizerV2Encoder, TokenizerV2EncoderOutput},
    tokenizer::v2::config::TokenizerV2Config,
};

/// The complete 12Hz tokenizer model.
///
/// Combines encoder (for encoding audio to codes) and decoder (for decoding codes to audio).
/// - For TTS, we primarily use the decoder to convert generated codes to audio.
/// - For voice cloning, we use the encoder to convert reference audio to codes.
///
/// # Architecture
///
/// The encoder uses the Mimi neural audio codec architecture:
/// ```text
/// Audio (24kHz) → SeaNet Encoder → Transformer → Downsample → Split RVQ → Codes (12.5Hz)
/// ```
///
/// The decoder uses a custom Qwen3-TTS architecture:
/// ```text
/// Codes → Split RVQ Decode → Conv → Transformer → Upsample → Vocoder → Audio (24kHz)
/// ```
#[derive(Debug)]
pub struct TokenizerV2 {
    pub encoder: Option<TokenizerV2Encoder>,
    pub decoder: TokenizerV2Decoder,
    pub config: TokenizerV2Config,
}

impl TokenizerV2 {
    /// Create a new tokenizer with decoder only (for TTS generation).
    ///
    /// Use `with_encoder()` or `new_full()` if you need audio encoding capability.
    pub fn new(config: TokenizerV2Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let decoder =
            TokenizerV2Decoder::new(&config.decoder_config, use_flash_attn, vb.pp("decoder"))?;
        Ok(Self {
            encoder: None,
            decoder,
            config,
        })
    }

    /// Create a new tokenizer with both encoder and decoder (for voice cloning).
    ///
    /// # Arguments
    /// * `config` - Tokenizer configuration
    /// * `use_flash_attn` - Whether to use flash attention (CUDA only)
    /// * `encoder_vb` - Variable builder for encoder weights (root of speech_tokenizer weights)
    /// * `decoder_vb` - Variable builder for decoder weights
    ///
    /// # Note
    /// This uses `for` which handles the Qwen3-TTS weight prefix structure
    /// where encoder weights are nested under "encoder." (e.g., `encoder.encoder.layers.*`).
    pub fn new_full(
        config: TokenizerV2Config,
        use_flash_attn: bool,
        encoder_vb: VarBuilder,
        decoder_vb: VarBuilder,
    ) -> Result<Self> {
        // Use for which handles the weight prefix mapping
        let encoder = TokenizerV2Encoder::for_v2(config.encoder_valid_num_quantizers, encoder_vb)?;
        let decoder = TokenizerV2Decoder::new(&config.decoder_config, use_flash_attn, decoder_vb)?;
        Ok(Self {
            encoder: Some(encoder),
            decoder,
            config,
        })
    }

    /// Add an encoder to an existing tokenizer.
    ///
    /// # Arguments
    /// * `encoder_vb` - Variable builder for encoder weights (root of speech_tokenizer weights)
    ///
    /// # Note
    /// This uses `for` which handles the Qwen3-TTS weight prefix structure
    /// where encoder weights are nested under "encoder." (e.g., `encoder.encoder.layers.*`).
    pub fn with_encoder(mut self, encoder_vb: VarBuilder) -> Result<Self> {
        // Use for which handles the weight prefix mapping
        let encoder =
            TokenizerV2Encoder::for_v2(self.config.encoder_valid_num_quantizers, encoder_vb)?;
        self.encoder = Some(encoder);
        Ok(self)
    }

    pub fn load(config: TokenizerV2Config, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        Self::new(config, use_flash_attn, vb)
    }

    /// Check if the encoder is available.
    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// Get the configuration.
    pub fn config(&self) -> &TokenizerV2Config {
        &self.config
    }

    /// Get the input sample rate.
    pub fn input_sample_rate(&self) -> usize {
        self.config.input_sample_rate
    }

    /// Get the output sample rate.
    pub fn output_sample_rate(&self) -> usize {
        self.config.output_sample_rate
    }

    /// Get the encode downsample rate (samples per code).
    pub fn encode_downsample_rate(&self) -> usize {
        self.config.encode_downsample_rate
    }

    /// Get the decode upsample rate (samples per code).
    pub fn decode_upsample_rate(&self) -> usize {
        self.config.decode_upsample_rate
    }

    /// Get the number of valid quantizers used by the encoder.
    pub fn encoder_valid_num_quantizers(&self) -> usize {
        self.config.encoder_valid_num_quantizers
    }

    /// Encode audio waveform to discrete codes.
    ///
    /// Requires encoder to be loaded (use `new_full()` or `with_encoder()`).
    ///
    /// # Arguments
    /// * `audio` - Input audio waveform of shape `(batch, samples)`
    ///
    /// # Returns
    /// * Discrete codes of shape `(batch, num_quantizers, seq_len)`
    ///
    /// # Errors
    /// * Returns error if encoder is not loaded
    ///
    /// # Example
    /// ```no_run
    /// use candle_core::{Device, Tensor};
    ///
    /// # fn main() -> candle_core::Result<()> {
    /// # let device = Device::Cpu;
    /// # // tokenizer would be loaded from model files
    /// # let mut tokenizer: qwen3_tts::audio::tokenizer::v2::TokenizerV2 = todo!();
    /// let audio = Tensor::randn(0f32, 1., (1, 48000), &device)?;  // 2 seconds at 24kHz
    /// let codes = tokenizer.encode(&audio)?;  // Shape: (1, num_quantizers, seq_len)
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode(&mut self, audio: &Tensor) -> Result<Tensor> {
        match &mut self.encoder {
            Some(encoder) => encoder.encode(audio),
            None => candle_core::bail!(
                "Encoder not loaded. Use new_full() or with_encoder() to load the encoder."
            ),
        }
    }

    /// Encode audio with padding mask for variable-length sequences.
    ///
    /// # Arguments
    /// * `audio` - Input audio waveform of shape `(batch, samples)`
    /// * `padding_mask` - Mask of shape `(batch, samples)` where 1 = valid, 0 = padding
    ///
    /// # Returns
    /// * Encoder output containing list of code tensors, each of shape `(seq_len_i, num_quantizers)`
    pub fn encode_with_mask(
        &mut self,
        audio: &Tensor,
        padding_mask: &Tensor,
    ) -> Result<TokenizerV2EncoderOutput> {
        match &mut self.encoder {
            Some(encoder) => {
                let codes = encoder.encode_with_mask(
                    audio,
                    padding_mask,
                    self.config.encode_downsample_rate,
                )?;
                Ok(TokenizerV2EncoderOutput::new(codes))
            }
            None => candle_core::bail!(
                "Encoder not loaded. Use new_full() or with_encoder() to load the encoder."
            ),
        }
    }

    /// Decode audio codes to waveform.
    ///
    /// # Arguments
    /// * `codes` - Audio codes of shape `(batch, seq_len, num_quantizers)`
    ///
    /// # Returns
    /// * Audio waveform of shape `(batch, samples)`
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // Transpose to (batch, num_quantizers, seq_len) for decoder
        let codes = codes.transpose(1, 2)?;
        self.decoder.forward(&codes)
    }

    /// Decode with chunking for long sequences.
    ///
    /// # Arguments
    /// * `codes` - Audio codes of shape `(batch, seq_len, num_quantizers)`
    /// * `chunk_size` - Number of code frames per chunk
    /// * `left_context_size` - Overlap between chunks for continuity
    ///
    /// # Returns
    /// * Audio waveform of shape `(batch, samples)`
    pub fn chunked_decode(
        &self,
        codes: &Tensor,
        chunk_size: usize,
        left_context_size: usize,
    ) -> Result<Tensor> {
        let codes = codes.transpose(1, 2)?;
        self.decoder
            .chunked_decode(&codes, chunk_size, left_context_size)
    }

    /// Reset the encoder's internal streaming state.
    pub fn reset_encoder_state(&mut self) {
        if let Some(encoder) = &mut self.encoder {
            encoder.reset_state();
        }
    }
}
