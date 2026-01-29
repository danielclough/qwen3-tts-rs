use std::path::PathBuf;

pub mod batch_items;
pub mod model_path;
pub mod output_path;
pub mod tokenizer;
pub mod voice_prompt;

/// File paths and I/O configuration.
#[derive(Debug, Clone, Default)]
pub struct IoArgs {
    /// Output WAV file path (for single text mode)
    pub output: PathBuf,

    /// Input file for batch processing (format detected from extension: .json or .txt)
    pub file: Option<PathBuf>,

    /// Output directory for batch mode (files named output_0.wav, output_1.wav, etc.)
    pub output_dir: Option<PathBuf>,

    /// Save voice clone prompt to file for reuse
    pub save_prompt: Option<PathBuf>,

    /// Load voice clone prompt from file (skips reference audio processing)
    pub load_prompt: Option<PathBuf>,

    /// Enable debug output (token encoding, embedding info, generation stats)
    pub debug: bool,

    /// Enable tracing output (debug logs)
    pub tracing: bool,
}

/// Model loading configuration.
#[derive(Debug, Clone, Default)]
pub struct ModelArgs {
    /// HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    /// If not specified, a default model is chosen based on the synthesis mode.
    pub model: Option<String>,

    /// Path to a local model directory (overrides model ID)
    pub model_path: Option<PathBuf>,

    /// Device to use (cpu, cuda, metal)
    pub device: String,

    /// Data type (f32, f16, bf16)
    pub dtype: String,
}

/// Generation/sampling parameters.
#[derive(Debug, Clone, Default)]
pub struct GenerationArgs {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Sampling temperature (higher = more random)
    pub temperature: Option<f64>,

    /// Top-k sampling parameter
    pub top_k: Option<usize>,

    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f64>,

    /// Repetition penalty (default: 1.05, higher = less repetition)
    pub repetition_penalty: Option<f64>,

    /// Random seed for reproducible generation
    pub seed: Option<u64>,

    /// Use greedy sampling (temperature=0, do_sample=false) for deterministic output
    pub greedy: bool,

    /// Subtalker sampling temperature (default: 0.9)
    pub subtalker_temperature: Option<f64>,

    /// Subtalker top-k sampling parameter (default: 50)
    pub subtalker_top_k: Option<usize>,

    /// Subtalker top-p sampling parameter (default: 1.0)
    pub subtalker_top_p: Option<f64>,

    /// Disable sampling for subtalker (use greedy decoding)
    pub no_subtalker_sample: bool,

    /// Estimate max-tokens from text length and use it for generation
    pub estimate_max_tokens: bool,
}

/// Per-item synthesis data (for batch or single item processing).
#[derive(Debug, Clone)]
pub struct SynthesisItem {
    /// Text to synthesize
    pub text: String,

    /// Language for this item
    pub language: String,

    /// Output WAV file path
    pub output: PathBuf,
}

/// Voice cloning parameters.
#[derive(Debug, Clone)]
pub struct VoiceCloneParams {
    /// Reference audio path (None if using load_prompt)
    pub ref_audio: Option<PathBuf>,

    /// Reference text (transcript of ref_audio)
    pub ref_text: Option<String>,

    /// Use x-vector only mode (no in-context learning)
    pub x_vector_only: bool,

    /// Save prompt on this item (typically first item only)
    pub save_prompt: bool,
}

/// Voice configuration for synthesis.
#[derive(Debug, Clone, Default)]
pub struct VoiceArgs {
    /// Language for synthesis (auto, english, chinese, japanese, korean, french, german, spanish)
    pub language: String,

    /// Text to synthesize
    pub text: Option<String>,

    /// Speaker name for CustomVoice model (e.g., vivian, Ethan, etc.)
    pub speaker: Option<String>,

    /// Instruction for the speaker (optional)
    pub instruct: Option<String>,

    /// Voice description for VoiceDesign model
    pub voice_design: Option<String>,

    /// Path or URL to reference audio for voice cloning
    pub ref_audio: Option<PathBuf>,

    /// Reference text (transcript of ref_audio)
    pub ref_text: Option<String>,

    /// Use x-vector only mode (no in-context learning)
    pub x_vector_only: bool,
}

#[derive(Debug, Clone)]
pub enum SynthesisMode {
    /// Synthesize using a predefined speaker (CustomVoice model)
    CustomVoice {
        /// Speaker name
        speaker: String,

        /// Optional instruction for the speaker
        instruct: Option<String>,
    },

    /// Synthesize using a voice description (VoiceDesign model)
    VoiceDesign {
        /// Natural language description of the voice
        description: String,
    },

    /// Clone a voice from reference audio (Base model)
    VoiceClone {
        /// Path to reference audio file
        audio: PathBuf,

        /// Transcript of the reference audio
        transcript: Option<String>,

        /// Use x-vector only (speaker embedding without ICL)
        x_vector_only: bool,
    },
}
