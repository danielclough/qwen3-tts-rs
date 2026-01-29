//! Unified tokenizer wrapper supporting both 12Hz and 25Hz tokenizers.

use candle_core::{Result, Tensor};

use crate::audio::{encoder::v2::TokenizerV2EncoderOutput, tokenizer::v2::TokenizerV2};
use crate::audio::tokenizer::v1::{TokenizerV1, TokenizerV1EncoderOutput};

/// Tokenizer type identifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenizerType {
    /// 12.5 Hz tokenizer (V2)
    Hz12,
    /// 25 Hz tokenizer (V1)
    Hz25,
}

/// Encoder output from either tokenizer variant.
pub enum EncoderOutput {
    V2(TokenizerV2EncoderOutput),
    V1(TokenizerV1EncoderOutput),
}

/// Decoder input for V1 (codes + conditioning).
pub struct V1DecodeInput<'a> {
    pub codes: &'a Tensor,
    pub xvectors: &'a Tensor,
    pub ref_mels: &'a Tensor,
}

/// Unified wrapper for audio tokenizers.
///
/// Provides a common interface for both 12Hz and 25Hz tokenizers.
pub enum TokenizerWrapper {
    /// 12Hz tokenizer (V2)
    V2(TokenizerV2),
    /// 25Hz tokenizer (V1)
    V1(TokenizerV1),
}

impl TokenizerWrapper {
    /// Create from 12Hz tokenizer.
    pub fn from_v2(tokenizer: TokenizerV2) -> Self {
        Self::V2(tokenizer)
    }

    /// Create from 25Hz tokenizer.
    pub fn from_v1(tokenizer: TokenizerV1) -> Self {
        Self::V1(tokenizer)
    }

    /// Get the tokenizer type.
    pub fn tokenizer_type(&self) -> TokenizerType {
        match self {
            Self::V2(_) => TokenizerType::Hz12,
            Self::V1(_) => TokenizerType::Hz25,
        }
    }

    /// Get the tokenizer rate in Hz.
    pub fn rate(&self) -> f64 {
        match self {
            Self::V2(t) => t.config().tokenizer_rate(),
            Self::V1(t) => t.config().tokenizer_rate(),
        }
    }

    /// Get the input sample rate.
    pub fn input_sample_rate(&self) -> usize {
        match self {
            Self::V2(t) => t.input_sample_rate(),
            Self::V1(t) => t.input_sample_rate(),
        }
    }

    /// Get the output sample rate.
    pub fn output_sample_rate(&self) -> usize {
        match self {
            Self::V2(t) => t.output_sample_rate(),
            Self::V1(t) => t.output_sample_rate(),
        }
    }

    /// Get the encode downsample rate (samples per code frame).
    pub fn encode_downsample_rate(&self) -> usize {
        match self {
            Self::V2(t) => t.encode_downsample_rate(),
            Self::V1(t) => t.encode_downsample_rate(),
        }
    }

    /// Get the decode upsample rate (samples per code frame).
    pub fn decode_upsample_rate(&self) -> usize {
        match self {
            Self::V2(t) => t.decode_upsample_rate(),
            Self::V1(t) => t.decode_upsample_rate(),
        }
    }

    /// Check if the encoder is available.
    pub fn has_encoder(&self) -> bool {
        match self {
            Self::V2(t) => t.has_encoder(),
            Self::V1(t) => t.has_encoder(),
        }
    }

    /// Encode audio waveform to discrete codes (V2 only).
    ///
    /// For V1, use `encode_v1()` which returns the full `TokenizerV1EncoderOutput`.
    pub fn encode(&mut self, audio: &Tensor) -> Result<Tensor> {
        match self {
            Self::V2(t) => t.encode(audio),
            Self::V1(_) => candle_core::bail!("V1 encode requires padding_mask; use encode_unified()"),
        }
    }

    /// Encode audio with padding mask.
    ///
    /// Returns a unified `EncoderOutput` enum.
    pub fn encode_unified(
        &mut self,
        audio: &Tensor,
        padding_mask: &Tensor,
    ) -> Result<EncoderOutput> {
        match self {
            Self::V2(t) => t.encode_with_mask(audio, padding_mask).map(EncoderOutput::V2),
            Self::V1(t) => t.encode(audio, padding_mask).map(EncoderOutput::V1),
        }
    }

    /// Encode audio with padding mask (V2-specific, returns V2 output type).
    pub fn encode_with_mask(
        &mut self,
        audio: &Tensor,
        padding_mask: &Tensor,
    ) -> Result<TokenizerV2EncoderOutput> {
        match self {
            Self::V2(t) => t.encode_with_mask(audio, padding_mask),
            Self::V1(_) => candle_core::bail!("V1 encoder returns TokenizerV1EncoderOutput; use encode_unified()"),
        }
    }

    /// Reset the encoder's internal streaming state.
    pub fn reset_encoder_state(&mut self) {
        match self {
            Self::V2(t) => t.reset_encoder_state(),
            Self::V1(_) => {} // V1 encoder is stateless
        }
    }

    /// Decode audio codes to waveform (V2 only).
    ///
    /// For V1, use `decode_v1()` which takes additional conditioning inputs.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        match self {
            Self::V2(t) => t.decode(codes),
            Self::V1(_) => candle_core::bail!("V1 decode requires xvectors and ref_mels; use decode_v1()"),
        }
    }

    /// Decode codes to audio (V1-specific).
    ///
    /// Returns a list of trimmed waveform tensors, one per batch item.
    pub fn decode_v1(
        &self,
        codes: &Tensor,
        xvectors: &Tensor,
        ref_mels: &Tensor,
    ) -> Result<Vec<Tensor>> {
        match self {
            Self::V1(t) => t.decode(codes, xvectors, ref_mels),
            Self::V2(_) => candle_core::bail!("decode_v1() is only available for V1 tokenizer"),
        }
    }

    /// Decode with chunking for long sequences (V2 only).
    ///
    /// V1 does not support chunked decoding (DiT processes full sequence).
    pub fn chunked_decode(
        &self,
        codes: &Tensor,
        chunk_size: usize,
        left_context_size: usize,
    ) -> Result<Tensor> {
        match self {
            Self::V2(t) => t.chunked_decode(codes, chunk_size, left_context_size),
            Self::V1(_) => candle_core::bail!("V1 does not support chunked decoding"),
        }
    }
}
