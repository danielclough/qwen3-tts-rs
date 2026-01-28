//! Text processing utilities for TTS.
//!
//! Provides tokenization and chat template formatting for Qwen3-TTS.
//! Uses the Qwen2 tokenizer vocabulary with special TTS tokens.
//!
//! Text is automatically normalized before tokenization to handle
//! typographic characters (smart quotes, em-dashes, etc.) that may
//! not be well-represented in the model's training data.
//!
//! # Batch Processing
//!
//! The processor supports batch tokenization with automatic left-padding:
//!
//! ```no_run
//! use qwen3_tts::text::processing::{TextProcessor, PaddingSide};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let processor = TextProcessor::from_file("tokenizer.json")?;
//! let texts = vec!["Hello", "World, how are you?"];
//! let output = processor.batch_tokenize_padded(&texts, PaddingSide::Left);
//! // output.input_ids: [[PAD, PAD, id1, id2], [id1, id2, id3, id4]]
//! // output.attention_mask: [[0, 0, 1, 1], [1, 1, 1, 1]]
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use tokenizers::Tokenizer;

/// Normalize text for consistent tokenization and better TTS output.
///
/// This ensures parity with Python's text handling, preventing degraded audio
/// quality when text contains smart/curly quotes, em-dashes, or other
/// typographic characters from word processors that may not be well-represented
/// in the model's training data.
///
/// # Normalizations
///
/// **Typographic characters:**
/// - Smart single quotes (`'` `'`) → ASCII apostrophe (`'`)
/// - Smart double quotes (`"` `"`) → ASCII quote (`"`)
/// - Em dash (`—`) → double hyphen (`--`)
/// - En dash (`–`) → hyphen (`-`)
/// - Horizontal ellipsis (`…`) → three periods (`...`)
///
/// **Whitespace:**
/// - Non-breaking space → regular space
/// - Narrow no-break space → regular space
/// - Ideographic space → regular space
/// - CRLF/CR → LF (Unix line endings)
///
/// **Zero-width characters (removed):**
/// - Zero-width space
/// - Zero-width non-joiner/joiner
/// - Byte order mark (BOM)
pub fn normalize_text(text: &str) -> String {
    text
        // Typographic quotes and punctuation
        .replace(['\u{2019}', '\u{2018}'], "'") // Smart single quotes → apostrophe
        .replace(['\u{201C}', '\u{201D}'], "\"") // Smart double quotes → quote
        .replace('\u{2014}', "--") // Em dash → double hyphen
        .replace('\u{2013}', "-") // En dash → hyphen
        .replace('\u{2026}', "...") // Horizontal ellipsis → three periods
        // Unicode whitespace → ASCII space
        .replace(['\u{00A0}', '\u{202F}', '\u{3000}'], " ")
        // Normalize line endings
        .replace("\r\n", "\n") // Windows CRLF → LF
        .replace('\r', "\n") // Old Mac CR → LF
        // Remove zero-width characters that break tokenization
        .replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Special token IDs used by Qwen3-TTS.
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokenIds {
    /// `<|im_start|>` token ID
    pub im_start: u32,
    /// `<|im_end|>` token ID
    pub im_end: u32,
    /// TTS padding token ID
    pub tts_pad: u32,
    /// TTS beginning-of-sequence token ID
    pub tts_bos: u32,
    /// TTS end-of-sequence token ID
    pub tts_eos: u32,
}

impl Default for SpecialTokenIds {
    fn default() -> Self {
        Self {
            im_start: 151644,
            im_end: 151645,
            tts_pad: 151671,
            tts_bos: 151672,
            tts_eos: 151673,
        }
    }
}

/// Padding side for batch tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingSide {
    /// Pad on the left (default for TTS, enables proper autoregressive generation).
    #[default]
    Left,
    /// Pad on the right (standard for most NLP tasks).
    Right,
}

/// Output from batch tokenization with padding.
#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    /// Token IDs for each sequence, padded to the same length.
    /// Shape: (batch_size, max_seq_len)
    pub input_ids: Vec<Vec<u32>>,
    /// Attention mask indicating real tokens (1) vs padding (0).
    /// Shape: (batch_size, max_seq_len)
    pub attention_mask: Vec<Vec<u32>>,
    /// Original sequence lengths before padding.
    pub lengths: Vec<usize>,
}

/// Text processor for preparing input to the TTS model.
///
/// Handles text tokenization using the Qwen2 tokenizer and provides
/// chat template formatting methods for different TTS modes.
#[derive(Debug, Clone)]
pub struct TextProcessor {
    /// The underlying tokenizer (Qwen2)
    tokenizer: Option<Tokenizer>,
    /// Special token IDs
    pub special_tokens: SpecialTokenIds,
}

impl TextProcessor {
    /// Create a new text processor without a loaded tokenizer.
    ///
    /// Use `load()` or `from_file()` to create a processor with a tokenizer.
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            special_tokens: SpecialTokenIds::default(),
        }
    }

    /// Create a text processor with custom special token IDs.
    pub fn with_special_tokens(special_tokens: SpecialTokenIds) -> Self {
        Self {
            tokenizer: None,
            special_tokens,
        }
    }

    /// Load tokenizer from a model directory.
    ///
    /// Tries multiple loading strategies in order:
    /// 1. `tokenizer.json` (HuggingFace fast tokenizer format)
    /// 2. `vocab.json` + `merges.txt` + `tokenizer_config.json` (BPE format with special tokens)
    ///
    /// # Arguments
    /// * `model_dir` - Path to the model directory
    ///
    /// # Example
    /// ```no_run
    /// use qwen3_tts::text::processing::TextProcessor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let processor = TextProcessor::from_pretrained("/path/to/model")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let model_dir = model_dir.as_ref();

        // Strategy 1: Try tokenizer.json (fast tokenizer format) - PREFERRED
        let tokenizer_json = model_dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            tracing::debug!("Loading tokenizer from tokenizer.json (HuggingFace fast format)");
            return Self::from_file(&tokenizer_json);
        }

        // Strategy 2: Try vocab.json + merges.txt + tokenizer_config.json (BPE format)
        let vocab_json = model_dir.join("vocab.json");
        let merges_txt = model_dir.join("merges.txt");
        let tokenizer_config = model_dir.join("tokenizer_config.json");
        if vocab_json.exists() {
            tracing::debug!(
                "Loading tokenizer from vocab.json + merges.txt + tokenizer_config.json"
            );
            return Self::from_bpe_files_with_config(
                &vocab_json,
                if merges_txt.exists() {
                    Some(&merges_txt)
                } else {
                    None
                },
                if tokenizer_config.exists() {
                    Some(&tokenizer_config)
                } else {
                    None
                },
            );
        }

        Err(format!(
            "No tokenizer found in {}. Expected tokenizer.json or vocab.json",
            model_dir.display()
        )
        .into())
    }

    /// Load tokenizer from vocab.json, merges.txt, and tokenizer_config.json.
    ///
    /// This replicates Python's Qwen2Tokenizer.__init__() from:
    /// transformers/src/transformers/models/qwen2/tokenization_qwen2.py
    ///
    /// The tokenizer is constructed programmatically with:
    /// - BPE model from vocab.json + merges.txt
    /// - NFC normalizer
    /// - Pre-tokenizer: Split(GPT-4 regex) + ByteLevel
    /// - Decoder: ByteLevel
    /// - Special tokens from tokenizer_config.json's added_tokens_decoder
    ///
    /// # Arguments
    /// * `vocab_path` - Path to vocab.json
    /// * `merges_path` - Optional path to merges.txt
    /// * `config_path` - Optional path to tokenizer_config.json (for special tokens)
    pub fn from_bpe_files_with_config(
        vocab_path: &Path,
        merges_path: Option<&Path>,
        config_path: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        use std::fs;
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::bpe::BPE;
        use tokenizers::normalizers::NFC;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::Split;
        use tokenizers::{AddedToken, SplitDelimiterBehavior, TokenizerImpl};

        tracing::debug!("Building Qwen2-style tokenizer (replicating tokenization_qwen2.py)");

        // Parse vocab.json -> AHashMap<String, u32>
        let vocab_content = fs::read_to_string(vocab_path)?;
        let vocab_std: std::collections::HashMap<String, u32> =
            serde_json::from_str(&vocab_content)?;
        let vocab: tokenizers::models::bpe::Vocab = vocab_std.into_iter().collect();
        tracing::debug!(entries = vocab.len(), "Loaded vocab");

        // Parse merges.txt -> Vec<(String, String)>
        let merges: tokenizers::models::bpe::Merges = if let Some(merges_path) = merges_path {
            let merges_content = fs::read_to_string(merges_path)?;
            merges_content
                .lines()
                .skip(1) // Skip header "#version: ..."
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split(' ').collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![]
        };
        tracing::debug!(count = merges.len(), "Loaded merges");

        // Parse tokenizer_config.json for special tokens
        let added_tokens: Vec<(u32, String, bool)> = if let Some(config_path) = config_path {
            let config_content = fs::read_to_string(config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_content)?;

            if let Some(added_tokens_decoder) = config
                .get("added_tokens_decoder")
                .and_then(|v| v.as_object())
            {
                let mut tokens: Vec<(u32, String, bool)> = added_tokens_decoder
                    .iter()
                    .filter_map(|(id_str, token_info)| {
                        let id: u32 = id_str.parse().ok()?;
                        let content = token_info.get("content")?.as_str()?.to_string();
                        let special = token_info
                            .get("special")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        Some((id, content, special))
                    })
                    .collect();
                tokens.sort_by_key(|(id, _, _)| *id);
                tracing::debug!(count = tokens.len(), "Found added tokens");
                tokens
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Build BPE model exactly like Python (tokenization_qwen2.py:61-72):
        // BPE(vocab=vocab, merges=merges, ...)
        let bpe = BPE::new(vocab, merges);

        // GPT-4 style regex pattern (from Python PRETOKENIZE_REGEX, tokenization_qwen2.py:33)
        let pretokenize_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        // Build pre-tokenizer exactly like Python (tokenization_qwen2.py:75-87):
        // pre_tokenizers.Sequence([
        //     pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
        //     pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        // ])
        let split = Split::new(pretokenize_regex, SplitDelimiterBehavior::Isolated, false)?;
        let byte_level = ByteLevel::new(false, true, false);
        let pre_tokenizer = Sequence::new(vec![split.into(), byte_level.into()]);

        // Assemble tokenizer like Python (tokenization_qwen2.py:61-87):
        // self._tokenizer = Tokenizer(BPE(...))
        // self._tokenizer.normalizer = normalizers.NFC()
        // self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([...])
        // self._tokenizer.decoder = decoders.ByteLevel()
        use tokenizers::{
            DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
        };
        type FullTokenizer = TokenizerImpl<
            BPE,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        >;
        let mut tokenizer: FullTokenizer = TokenizerImpl::new(bpe);
        tokenizer.with_normalizer(Some(NFC));
        tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, true, true)));

        // Convert to Tokenizer (the wrapper type we use)
        let mut tokenizer: Tokenizer = tokenizer.into();

        // Add special tokens from tokenizer_config.json (tokenization_qwen2.py:98):
        // self.add_tokens([AddedToken(token, special=True) for token in self.all_special_tokens])
        if !added_tokens.is_empty() {
            let special_tokens: Vec<AddedToken> = added_tokens
                .iter()
                .filter(|(_, _, special)| *special)
                .map(|(_, content, _)| {
                    AddedToken::from(content.clone(), true)
                        .single_word(false)
                        .lstrip(false)
                        .rstrip(false)
                        .normalized(false)
                })
                .collect();

            if !special_tokens.is_empty() {
                tracing::debug!(count = special_tokens.len(), "Adding special tokens");
                tokenizer.add_special_tokens(&special_tokens);
            }
        }

        tracing::debug!("Tokenizer built successfully");

        Ok(Self {
            tokenizer: Some(tokenizer),
            special_tokens: SpecialTokenIds::default(),
        })
    }

    /// Load tokenizer from vocab.json and optional merges.txt (BPE format).
    ///
    /// DEPRECATED: Use from_bpe_files_with_config instead to properly handle special tokens.
    ///
    /// # Arguments
    /// * `vocab_path` - Path to vocab.json
    /// * `merges_path` - Optional path to merges.txt
    pub fn from_bpe_files<P: AsRef<Path>>(
        vocab_path: P,
        merges_path: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::from_bpe_files_with_config(vocab_path.as_ref(), merges_path, None)
    }

    /// Load tokenizer from a HuggingFace tokenizer.json file.
    ///
    /// # Arguments
    /// * `path` - Path to the tokenizer.json file
    ///
    /// # Example
    /// ```no_run
    /// use qwen3_tts::text::processing::TextProcessor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let processor = TextProcessor::from_file("path/to/tokenizer.json")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = Tokenizer::from_file(path)?;
        Ok(Self {
            tokenizer: Some(tokenizer),
            special_tokens: SpecialTokenIds::default(),
        })
    }

    /// Load tokenizer from bytes (tokenizer.json content).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = Tokenizer::from_bytes(bytes)?;
        Ok(Self {
            tokenizer: Some(tokenizer),
            special_tokens: SpecialTokenIds::default(),
        })
    }

    /// Check if a tokenizer is loaded.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Build assistant text with chat template.
    ///
    /// Format: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
    ///
    /// This is used for the main text-to-speech input.
    pub fn build_assistant_text(&self, text: &str) -> String {
        format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            text
        )
    }

    /// Build reference text with chat template.
    ///
    /// Format: `<|im_start|>assistant\n{text}<|im_end|>\n`
    ///
    /// This is used for reference text in voice cloning ICL mode.
    pub fn build_ref_text(&self, text: &str) -> String {
        format!("<|im_start|>assistant\n{}<|im_end|>\n", text)
    }

    /// Build instruction text with chat template.
    ///
    /// Format: `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// This is used for voice design instructions.
    pub fn build_instruct_text(&self, instruct: &str) -> String {
        format!("<|im_start|>user\n{}<|im_end|>\n", instruct)
    }

    /// Tokenize text into token IDs.
    ///
    /// Text is automatically normalized before tokenization to handle
    /// typographic characters (smart quotes, em-dashes, etc.).
    ///
    /// # Arguments
    /// * `text` - The text to tokenize
    ///
    /// # Returns
    /// Vector of token IDs, or empty vector if no tokenizer is loaded.
    ///
    /// # Panics
    /// Panics if the tokenizer fails to encode the text.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        // Normalize typographic characters before tokenization
        let normalized = normalize_text(text);
        match &self.tokenizer {
            Some(tokenizer) => {
                let encoding = tokenizer
                    .encode(normalized.as_str(), false)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to encode text: {:?}\n\
                            This may indicate that special tokens like <|im_start|> or <|im_end|> \
                            are not properly defined in the tokenizer. \
                            Ensure you're using a valid tokenizer.json from the Qwen model.\n\
                            Text being tokenized (first 200 chars): {:?}",
                            e,
                            &text.chars().take(200).collect::<String>()
                        )
                    });
                encoding.get_ids().to_vec()
            }
            None => {
                tracing::warn!("No tokenizer loaded, returning empty token list");
                vec![]
            }
        }
    }

    /// Tokenize text with error handling.
    ///
    /// Text is automatically normalized before tokenization to handle
    /// typographic characters (smart quotes, em-dashes, etc.).
    ///
    /// # Arguments
    /// * `text` - The text to tokenize
    ///
    /// # Returns
    /// Result containing token IDs or an error.
    pub fn try_tokenize(
        &self,
        text: &str,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        // Normalize typographic characters before tokenization
        let normalized = normalize_text(text);
        match &self.tokenizer {
            Some(tokenizer) => {
                let encoding = tokenizer.encode(normalized.as_str(), false)?;
                Ok(encoding.get_ids().to_vec())
            }
            None => Err("No tokenizer loaded".into()),
        }
    }

    /// Tokenize text for TTS with chat template formatting.
    ///
    /// Wraps the text in the assistant chat template before tokenizing.
    ///
    /// # Arguments
    /// * `text` - The text to convert to speech
    ///
    /// # Returns
    /// Vector of token IDs ready for the TTS model.
    pub fn tokenize_for_tts(&self, text: &str) -> Vec<u32> {
        let formatted = self.build_assistant_text(text);
        let tokens = self.tokenize(&formatted);

        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(
                input_text = %formatted,
                token_count = tokens.len(),
                token_ids = ?tokens,
                "tokenize_for_tts"
            );
            if let Some(ref tokenizer) = self.tokenizer {
                let decoded_tokens: Vec<String> = tokens
                    .iter()
                    .map(|&id| {
                        tokenizer
                            .decode(&[id], true)
                            .unwrap_or_else(|_| format!("<{}>", id))
                    })
                    .collect();
                tracing::debug!(decoded_tokens = ?decoded_tokens, "tokenize_for_tts decoded");
            }
        }

        tokens
    }

    /// Tokenize reference text for voice cloning.
    ///
    /// # Arguments
    /// * `text` - The reference text
    ///
    /// # Returns
    /// Vector of token IDs for the reference text.
    pub fn tokenize_ref_text(&self, text: &str) -> Vec<u32> {
        let formatted = self.build_ref_text(text);
        self.tokenize(&formatted)
    }

    /// Tokenize instruction for voice design.
    ///
    /// # Arguments
    /// * `instruct` - The voice design instruction
    ///
    /// # Returns
    /// Vector of token IDs for the instruction.
    pub fn tokenize_instruct(&self, instruct: &str) -> Vec<u32> {
        let formatted = self.build_instruct_text(instruct);
        self.tokenize(&formatted)
    }

    /// Add TTS special tokens to a token sequence.
    ///
    /// # Arguments
    /// * `token_ids` - The token IDs to wrap
    ///
    /// # Returns
    /// Token IDs with TTS BOS prepended.
    pub fn add_tts_tokens(&self, token_ids: Vec<u32>) -> Vec<u32> {
        let mut result = vec![self.special_tokens.tts_bos];
        result.extend(token_ids);
        result
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    /// * `token_ids` - The token IDs to decode
    ///
    /// # Returns
    /// The decoded text, or None if no tokenizer is loaded.
    pub fn decode(&self, token_ids: &[u32]) -> Option<String> {
        self.tokenizer
            .as_ref()
            .map(|tokenizer| tokenizer.decode(token_ids, true).unwrap_or_default())
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> Option<usize> {
        self.tokenizer.as_ref().map(|t| t.get_vocab_size(true))
    }

    // =========================================================================
    // Batch Tokenization Methods
    // =========================================================================

    /// Tokenize multiple texts into token ID sequences.
    ///
    /// # Arguments
    /// * `texts` - The texts to tokenize
    ///
    /// # Returns
    /// Vector of token ID vectors for each text.
    pub fn batch_tokenize(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }

    /// Tokenize multiple texts with error handling.
    ///
    /// # Arguments
    /// * `texts` - The texts to tokenize
    ///
    /// # Returns
    /// Result containing token IDs for each text or an error.
    pub fn try_batch_tokenize(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error + Send + Sync>> {
        if self.tokenizer.is_none() {
            return Err("No tokenizer loaded".into());
        }
        texts.iter().map(|text| self.try_tokenize(text)).collect()
    }

    /// Tokenize multiple texts with automatic padding.
    ///
    /// Sequences are padded to the maximum length in the batch.
    /// Default padding uses the TTS pad token (151671).
    ///
    /// # Arguments
    /// * `texts` - The texts to tokenize
    /// * `padding_side` - Whether to pad on the left (default) or right
    ///
    /// # Returns
    /// `TokenizerOutput` with padded input_ids, attention_mask, and original lengths.
    pub fn batch_tokenize_padded(
        &self,
        texts: &[&str],
        padding_side: PaddingSide,
    ) -> Result<TokenizerOutput, Box<dyn std::error::Error + Send + Sync>> {
        self.batch_tokenize_padded_with_pad_token(texts, padding_side, self.special_tokens.tts_pad)
    }

    /// Tokenize multiple texts with automatic padding using a custom pad token.
    ///
    /// # Arguments
    /// * `texts` - The texts to tokenize
    /// * `padding_side` - Whether to pad on the left or right
    /// * `pad_token_id` - The token ID to use for padding
    ///
    /// # Returns
    /// `TokenizerOutput` with padded input_ids, attention_mask, and original lengths.
    pub fn batch_tokenize_padded_with_pad_token(
        &self,
        texts: &[&str],
        padding_side: PaddingSide,
        pad_token_id: u32,
    ) -> Result<TokenizerOutput, Box<dyn std::error::Error + Send + Sync>> {
        let sequences = self.try_batch_tokenize(texts)?;
        Ok(Self::pad_sequences(&sequences, padding_side, pad_token_id))
    }

    /// Tokenize texts for TTS with chat template and padding.
    ///
    /// Each text is wrapped in the assistant chat template before tokenizing,
    /// then all sequences are padded to the same length.
    ///
    /// # Arguments
    /// * `texts` - The texts to convert to speech
    /// * `padding_side` - Whether to pad on the left (default) or right
    ///
    /// # Returns
    /// `TokenizerOutput` with padded input_ids ready for the TTS model.
    pub fn batch_tokenize_for_tts(
        &self,
        texts: &[&str],
        padding_side: PaddingSide,
    ) -> Result<TokenizerOutput, Box<dyn std::error::Error + Send + Sync>> {
        let formatted: Vec<String> = texts.iter().map(|t| self.build_assistant_text(t)).collect();
        let formatted_refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();
        self.batch_tokenize_padded(&formatted_refs, padding_side)
    }

    /// Pad a batch of token sequences to the same length.
    ///
    /// # Arguments
    /// * `sequences` - The token sequences to pad
    /// * `padding_side` - Whether to pad on the left or right
    /// * `pad_token_id` - The token ID to use for padding
    ///
    /// # Returns
    /// `TokenizerOutput` with padded sequences and attention masks.
    pub fn pad_sequences(
        sequences: &[Vec<u32>],
        padding_side: PaddingSide,
        pad_token_id: u32,
    ) -> TokenizerOutput {
        if sequences.is_empty() {
            return TokenizerOutput {
                input_ids: vec![],
                attention_mask: vec![],
                lengths: vec![],
            };
        }

        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let max_len = lengths.iter().copied().max().unwrap_or(0);

        let mut input_ids = Vec::with_capacity(sequences.len());
        let mut attention_mask = Vec::with_capacity(sequences.len());

        for seq in sequences {
            let pad_len = max_len - seq.len();
            let mut padded_ids = Vec::with_capacity(max_len);
            let mut mask = Vec::with_capacity(max_len);

            match padding_side {
                PaddingSide::Left => {
                    // Add padding on the left
                    padded_ids.extend(std::iter::repeat_n(pad_token_id, pad_len));
                    padded_ids.extend(seq.iter().copied());
                    // Mask: 0 for padding, 1 for real tokens
                    mask.extend(std::iter::repeat_n(0u32, pad_len));
                    mask.extend(std::iter::repeat_n(1u32, seq.len()));
                }
                PaddingSide::Right => {
                    // Add padding on the right
                    padded_ids.extend(seq.iter().copied());
                    padded_ids.extend(std::iter::repeat_n(pad_token_id, pad_len));
                    // Mask: 1 for real tokens, 0 for padding
                    mask.extend(std::iter::repeat_n(1u32, seq.len()));
                    mask.extend(std::iter::repeat_n(0u32, pad_len));
                }
            }

            input_ids.push(padded_ids);
            attention_mask.push(mask);
        }

        TokenizerOutput {
            input_ids,
            attention_mask,
            lengths,
        }
    }

    /// Decode multiple token sequences back to text.
    ///
    /// # Arguments
    /// * `batch_ids` - The batch of token ID sequences to decode
    ///
    /// # Returns
    /// Vector of decoded strings, or empty vector if no tokenizer is loaded.
    pub fn batch_decode(&self, batch_ids: &[&[u32]]) -> Vec<Option<String>> {
        batch_ids.iter().map(|ids| self.decode(ids)).collect()
    }

    /// Decode multiple token sequences, skipping special tokens.
    ///
    /// # Arguments
    /// * `batch_ids` - The batch of token ID sequences to decode
    /// * `skip_special_tokens` - Whether to skip special tokens in decoding
    ///
    /// # Returns
    /// Vector of decoded strings.
    pub fn batch_decode_with_options(
        &self,
        batch_ids: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Vec<Option<String>> {
        match &self.tokenizer {
            Some(tokenizer) => batch_ids
                .iter()
                .map(|ids| tokenizer.decode(ids, skip_special_tokens).ok())
                .collect(),
            None => batch_ids.iter().map(|_| None).collect(),
        }
    }
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        // Smart quotes
        assert_eq!(normalize_text("\u{2018}hello\u{2019}"), "'hello'");
        assert_eq!(normalize_text("\u{201C}world\u{201D}"), "\"world\"");

        // Dashes
        assert_eq!(normalize_text("one\u{2013}two"), "one-two"); // en-dash
        assert_eq!(normalize_text("one\u{2014}two"), "one--two"); // em-dash

        // Ellipsis
        assert_eq!(normalize_text("wait\u{2026}"), "wait...");

        // Unicode whitespace
        assert_eq!(normalize_text("hello\u{00A0}world"), "hello world"); // NBSP
        assert_eq!(normalize_text("hello\u{202F}world"), "hello world"); // narrow NBSP
        assert_eq!(normalize_text("hello\u{3000}world"), "hello world"); // ideographic space

        // Line endings
        assert_eq!(normalize_text("a\r\nb"), "a\nb"); // CRLF
        assert_eq!(normalize_text("a\rb"), "a\nb"); // CR

        // Zero-width characters (removed)
        assert_eq!(normalize_text("hel\u{200B}lo"), "hello"); // ZWSP
        assert_eq!(normalize_text("hel\u{FEFF}lo"), "hello"); // BOM
        assert_eq!(normalize_text("hel\u{200C}lo"), "hello"); // ZWNJ
        assert_eq!(normalize_text("hel\u{200D}lo"), "hello"); // ZWJ

        // Combined typographic
        assert_eq!(
            normalize_text("He said, \u{201C}It\u{2019}s\u{2014}well\u{2026}\u{201D}"),
            "He said, \"It's--well...\""
        );

        // ASCII passthrough
        assert_eq!(normalize_text("Hello, world!"), "Hello, world!");
    }

    #[test]
    fn test_chat_templates() {
        let processor = TextProcessor::new();

        let assistant = processor.build_assistant_text("Hello world");
        assert_eq!(
            assistant,
            "<|im_start|>assistant\nHello world<|im_end|>\n<|im_start|>assistant\n"
        );

        let ref_text = processor.build_ref_text("Reference");
        assert_eq!(ref_text, "<|im_start|>assistant\nReference<|im_end|>\n");

        let instruct = processor.build_instruct_text("Speak slowly");
        assert_eq!(instruct, "<|im_start|>user\nSpeak slowly<|im_end|>\n");
    }

    #[test]
    fn test_special_token_defaults() {
        let tokens = SpecialTokenIds::default();
        assert_eq!(tokens.im_start, 151644);
        assert_eq!(tokens.im_end, 151645);
        assert_eq!(tokens.tts_bos, 151672);
    }

    #[test]
    fn test_tokenize_without_tokenizer() {
        let processor = TextProcessor::new();
        let tokens = processor.tokenize("Hello");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_add_tts_tokens() {
        let processor = TextProcessor::new();
        let tokens = vec![1, 2, 3];
        let with_tts = processor.add_tts_tokens(tokens);
        assert_eq!(with_tts, vec![151672, 1, 2, 3]);
    }

    #[test]
    fn test_padding_side_default() {
        let side = PaddingSide::default();
        assert_eq!(side, PaddingSide::Left);
    }

    #[test]
    fn test_pad_sequences_left() {
        // Test left padding with variable length sequences
        let sequences = vec![vec![1, 2], vec![3, 4, 5, 6], vec![7]];
        let output = TextProcessor::pad_sequences(&sequences, PaddingSide::Left, 0);

        // Max length is 4, all should be padded to 4
        assert_eq!(output.input_ids.len(), 3);
        assert_eq!(output.input_ids[0], vec![0, 0, 1, 2]); // 2 pads on left
        assert_eq!(output.input_ids[1], vec![3, 4, 5, 6]); // no padding
        assert_eq!(output.input_ids[2], vec![0, 0, 0, 7]); // 3 pads on left

        // Attention masks
        assert_eq!(output.attention_mask[0], vec![0, 0, 1, 1]);
        assert_eq!(output.attention_mask[1], vec![1, 1, 1, 1]);
        assert_eq!(output.attention_mask[2], vec![0, 0, 0, 1]);

        // Original lengths
        assert_eq!(output.lengths, vec![2, 4, 1]);
    }

    #[test]
    fn test_pad_sequences_right() {
        let sequences = vec![vec![1, 2], vec![3, 4, 5]];
        let output = TextProcessor::pad_sequences(&sequences, PaddingSide::Right, 99);

        assert_eq!(output.input_ids[0], vec![1, 2, 99]); // 1 pad on right
        assert_eq!(output.input_ids[1], vec![3, 4, 5]); // no padding

        assert_eq!(output.attention_mask[0], vec![1, 1, 0]);
        assert_eq!(output.attention_mask[1], vec![1, 1, 1]);
    }

    #[test]
    fn test_pad_sequences_empty() {
        let sequences: Vec<Vec<u32>> = vec![];
        let output = TextProcessor::pad_sequences(&sequences, PaddingSide::Left, 0);
        assert!(output.input_ids.is_empty());
        assert!(output.attention_mask.is_empty());
        assert!(output.lengths.is_empty());
    }

    #[test]
    fn test_pad_sequences_single() {
        let sequences = vec![vec![1, 2, 3]];
        let output = TextProcessor::pad_sequences(&sequences, PaddingSide::Left, 0);

        assert_eq!(output.input_ids.len(), 1);
        assert_eq!(output.input_ids[0], vec![1, 2, 3]); // no padding needed
        assert_eq!(output.attention_mask[0], vec![1, 1, 1]);
        assert_eq!(output.lengths, vec![3]);
    }

    #[test]
    fn test_batch_tokenize_without_tokenizer() {
        let processor = TextProcessor::new();
        let results = processor.batch_tokenize(&["Hello", "World"]);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_empty());
        assert!(results[1].is_empty());
    }

    #[test]
    fn test_try_batch_tokenize_without_tokenizer() {
        let processor = TextProcessor::new();
        let result = processor.try_batch_tokenize(&["Hello", "World"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_decode_without_tokenizer() {
        let processor = TextProcessor::new();
        let ids: &[&[u32]] = &[&[1, 2, 3], &[4, 5]];
        let results = processor.batch_decode(ids);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_none());
        assert!(results[1].is_none());
    }

    #[test]
    fn test_tokenizer_output_structure() {
        let output = TokenizerOutput {
            input_ids: vec![vec![1, 2], vec![3, 4]],
            attention_mask: vec![vec![1, 1], vec![1, 1]],
            lengths: vec![2, 2],
        };
        assert_eq!(output.input_ids.len(), 2);
        assert_eq!(output.attention_mask.len(), 2);
        assert_eq!(output.lengths.len(), 2);
    }
}
