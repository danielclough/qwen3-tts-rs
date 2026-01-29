use clap::{Parser, Subcommand};
use qwen3_tts::io::{GenerationArgs, IoArgs, ModelArgs, SynthesisMode, VoiceArgs};
use std::path::PathBuf;

fn default_device() -> String {
    if cfg!(target_os = "macos") {
        "metal".to_string()
    } else {
        "cuda".to_string()
    }
}

/// Qwen3-TTS Command Line Interface
///
/// Generate speech from text using the Qwen3-TTS model.
/// Models are automatically downloaded from HuggingFace Hub.
#[derive(Parser, Debug)]
#[command(name = "qwen3-tts")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    #[arg(short = 'M', long)]
    pub model: Option<String>,

    /// Path to a local model directory (overrides --model)
    #[arg(short = 'p', long)]
    pub model_path: Option<PathBuf>,

    /// Text to synthesize (required for synthesis unless using --file or a subcommand)
    #[arg(short, long, required_unless_present = "file")]
    pub text: Option<String>,

    /// Output WAV file path (for single text mode)
    #[arg(short, long, default_value = "output.wav")]
    pub output: PathBuf,

    /// Input file for batch processing (format detected from extension: .json or .txt)
    #[arg(short, long, conflicts_with = "text")]
    pub file: Option<PathBuf>,

    /// Output directory for batch mode
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// Save voice clone prompt to file for reuse
    #[arg(long)]
    pub save_prompt: Option<PathBuf>,

    /// Load voice clone prompt from file (skips reference audio processing)
    #[arg(long, conflicts_with_all = ["ref_audio", "save_prompt"])]
    pub load_prompt: Option<PathBuf>,

    /// Language for synthesis
    #[arg(short, long, default_value = "auto")]
    pub language: String,

    /// Device to use (cpu, cuda, metal).
    #[arg(long, default_value_t = default_device())]
    pub device: String,

    /// Data type (f32, f16, bf16)
    #[arg(long, default_value = "bf16")]
    pub dtype: String,

    /// Model type subcommand
    #[command(subcommand)]
    pub mode: Option<CliSynthesisMode>,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "2048")]
    pub max_tokens: usize,

    /// Sampling temperature (higher = more random)
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Top-k sampling parameter
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Top-p (nucleus) sampling parameter
    #[arg(long)]
    pub top_p: Option<f64>,

    /// Repetition penalty
    #[arg(long)]
    pub repetition_penalty: Option<f64>,

    /// Random seed for reproducible generation
    #[arg(long)]
    pub seed: Option<u64>,

    /// Use greedy sampling for deterministic output
    #[arg(long)]
    pub greedy: bool,

    /// Subtalker sampling temperature
    #[arg(long)]
    pub subtalker_temperature: Option<f64>,

    /// Subtalker top-k sampling parameter
    #[arg(long)]
    pub subtalker_top_k: Option<usize>,

    /// Subtalker top-p sampling parameter
    #[arg(long)]
    pub subtalker_top_p: Option<f64>,

    /// Disable sampling for subtalker
    #[arg(long)]
    pub no_subtalker_sample: bool,

    /// Speaker name for CustomVoice model
    #[arg(long, conflicts_with_all = ["ref_audio", "ref_text", "voice_design"])]
    pub speaker: Option<String>,

    /// Instruction for the speaker
    #[arg(long)]
    pub instruct: Option<String>,

    /// Voice description for VoiceDesign model
    #[arg(long, conflicts_with_all = ["speaker", "ref_audio", "ref_text"])]
    pub voice_design: Option<String>,

    /// Path or URL to reference audio for voice cloning
    #[arg(long, conflicts_with_all = ["speaker", "voice_design"])]
    pub ref_audio: Option<PathBuf>,

    /// Reference text (transcript of ref_audio)
    #[arg(long, conflicts_with_all = ["speaker", "voice_design"])]
    pub ref_text: Option<String>,

    /// Use x-vector only mode (no in-context learning)
    #[arg(long)]
    pub x_vector_only: bool,

    /// Enable debug output
    #[arg(long)]
    pub debug: bool,

    /// Estimate max-tokens from text length and use it for generation
    #[arg(long)]
    pub estimate_max_tokens: bool,

    /// Enable tracing output (debug logs). Use RUST_LOG env var to filter levels.
    #[arg(long)]
    pub tracing: bool,

    /// Use flash attention (requires CUDA and flash-attn feature)
    #[arg(long)]
    pub flash_attn: bool,
}

#[derive(Subcommand, Debug, Clone)]
pub enum CliSynthesisMode {
    /// Synthesize using a predefined speaker (CustomVoice model)
    CustomVoice {
        #[arg(short, long)]
        speaker: String,
        #[arg(short, long)]
        instruct: Option<String>,
    },
    /// Synthesize using a voice description (VoiceDesign model)
    VoiceDesign {
        #[arg(short, long)]
        description: String,
    },
    /// Clone a voice from reference audio (Base model)
    VoiceClone {
        #[arg(short, long)]
        audio: PathBuf,
        #[arg(short, long)]
        transcript: Option<String>,
        #[arg(long)]
        x_vector_only: bool,
    },
    /// Audio tokenizer utilities (encode/decode audio)
    Tokenizer {
        #[command(subcommand)]
        command: TokenizerCommand,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum TokenizerCommand {
    /// Encode audio file to codes
    Encode {
        /// Input audio file (wav, mp3, flac, etc.)
        #[arg(short, long)]
        input: PathBuf,
        /// Output codes file (JSON)
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Decode codes back to audio
    Decode {
        /// Input codes file (JSON)
        #[arg(short, long)]
        input: PathBuf,
        /// Output audio file (WAV)
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Round-trip test: encode then decode
    Roundtrip {
        /// Input audio file
        #[arg(short, long)]
        input: PathBuf,
        /// Output reconstructed audio file (WAV)
        #[arg(short, long)]
        output: PathBuf,
        /// Also save intermediate codes to this file
        #[arg(long)]
        save_codes: Option<PathBuf>,
    },
}

impl Cli {
    pub fn to_io_args(&self) -> IoArgs {
        IoArgs {
            output: self.output.clone(),
            file: self.file.clone(),
            output_dir: self.output_dir.clone(),
            save_prompt: self.save_prompt.clone(),
            load_prompt: self.load_prompt.clone(),
            debug: self.debug,
            tracing: self.tracing,
        }
    }

    pub fn to_model_args(&self) -> ModelArgs {
        ModelArgs {
            model: self.model.clone(),
            model_path: self.model_path.clone(),
            device: self.device.clone(),
            dtype: self.dtype.clone(),
        }
    }

    pub fn to_generation_args(&self) -> GenerationArgs {
        GenerationArgs {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            greedy: self.greedy,
            subtalker_temperature: self.subtalker_temperature,
            subtalker_top_k: self.subtalker_top_k,
            subtalker_top_p: self.subtalker_top_p,
            no_subtalker_sample: self.no_subtalker_sample,
            estimate_max_tokens: self.estimate_max_tokens,
        }
    }

    pub fn to_voice_args(&self) -> VoiceArgs {
        VoiceArgs {
            language: self.language.clone(),
            text: self.text.clone(),
            speaker: self.speaker.clone(),
            instruct: self.instruct.clone(),
            voice_design: self.voice_design.clone(),
            ref_audio: self.ref_audio.clone(),
            ref_text: self.ref_text.clone(),
            x_vector_only: self.x_vector_only,
        }
    }

    pub fn to_synthesis_mode(&self) -> Option<SynthesisMode> {
        self.mode.as_ref().and_then(|m| match m {
            CliSynthesisMode::CustomVoice { speaker, instruct } => {
                Some(SynthesisMode::CustomVoice {
                    speaker: speaker.clone(),
                    instruct: instruct.clone(),
                })
            }
            CliSynthesisMode::VoiceDesign { description } => Some(SynthesisMode::VoiceDesign {
                description: description.clone(),
            }),
            CliSynthesisMode::VoiceClone {
                audio,
                transcript,
                x_vector_only,
            } => Some(SynthesisMode::VoiceClone {
                audio: audio.clone(),
                transcript: transcript.clone(),
                x_vector_only: *x_vector_only,
            }),
            CliSynthesisMode::Tokenizer { .. } => None, // Not a synthesis mode
        })
    }
}
