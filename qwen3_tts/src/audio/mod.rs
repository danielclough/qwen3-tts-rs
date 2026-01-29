//! Audio processing components.
//!
//! This module contains:
//! - `decoder`: Audio decoder (codes → waveform)
//! - `encoder`: Audio encoder (waveform → codes)
//! - `tokenizer`: High-level tokenizer combining encoder/decoder
//! - `mel`: Mel spectrogram computation
//! - `utils`: Audio I/O utilities

pub mod decoder;
pub mod encoder;
pub mod mel;
pub mod mel_v1;
pub mod quantizer;
pub mod quantizer_v1;
pub mod tokenizer;
pub mod utils;
