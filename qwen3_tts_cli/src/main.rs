//! Command-line interface for Qwen3-TTS text-to-speech synthesis.
//!
//! This CLI supports all three model types:
//! - Base: Voice cloning from reference audio
//! - CustomVoice: Predefined speaker voices
//! - VoiceDesign: Voice description via natural language
//!
//! # Usage
//!
//! ```bash
//! # Build the CLI (with audio-loading feature for voice cloning)
//! cargo build --release -p qwen3_tts_cli --features audio-loading
//!
//! # Use default model (CustomVoice) - downloads automatically from HuggingFace
//! qwen3-tts --text "Hello, world!" --speaker vivian
//!
//! # Specify a HuggingFace model (larger 1.7B model)
//! qwen3-tts --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
//!     --text "Hello, world!" --speaker vivian
//!
//! # Use a local model path
//! qwen3-tts --model-path /path/to/model \
//!     --text "Hello, world!" --speaker vivian
//!
//! # Voice design synthesis
//! qwen3-tts --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
//!     --text "Hello, world!" \
//!     --voice-design "A warm female voice with a slight British accent"
//!
//! # Voice cloning (base model)
//! qwen3-tts --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
//!     --text "Hello, world!" \
//!     --ref-audio reference.wav --ref-text "This is the reference text"
//!
//! # Custom generation parameters (talker + subtalker)
//! qwen3-tts --text "Hello, world!" --speaker vivian \
//!     --temperature 0.9 --top-k 50 --top-p 1.0 \
//!     --repetition-penalty 1.05 \
//!     --subtalker-temperature 0.9 --subtalker-top-k 50 --subtalker-top-p 1.0
//!
//! # Greedy (deterministic) generation
//! qwen3-tts --text "Hello, world!" --speaker vivian --greedy
//!
//! # Batch processing (format detected from extension: .json or .txt)
//! qwen3-tts --file inputs.json --output-dir ./outputs/ --speaker vivian
//!
//! # Batch processing with text file (one line per item)
//! qwen3-tts --file texts.txt --output-dir ./outputs/ --speaker vivian
//!
//! # Save voice clone prompt for reuse
//! qwen3-tts --ref-audio reference.wav --ref-text "Reference transcript" \
//!     --text "Hello" --save-prompt voice_prompt.json
//!
//! # Reuse saved voice prompt (faster for batch processing)
//! qwen3-tts --load-prompt voice_prompt.json \
//!     --file inputs.json --output-dir ./outputs/
//! ```
//!
//! # Batch Input JSON Format
//!
//! ```json
//! {
//!   "items": [
//!     {"text": "Hello, world!", "language": "english"},
//!     {"text": "你好，世界！", "language": "chinese"},
//!     {"text": "Custom output", "output": "custom.wav"}
//!   ]
//! }
//! ```

mod args;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use clap::Parser;
use qwen3_tts::io::batch_items::load_batch_items;
use qwen3_tts::io::model_path::get_model_path;
use qwen3_tts::io::output_path::get_output_path;
use qwen3_tts::model::loader::{LoaderConfig, ModelLoader};
use qwen3_tts::nn::mt_rng::set_seed;
use qwen3_tts::synthesis::detect_mode::{DetectedMode, determine_mode};
use qwen3_tts::synthesis::synthesize_voice::{
    synthesize_custom_voice_item, synthesize_voice_clone_item, synthesize_voice_design_item,
};
use qwen3_tts::synthesis::tokenizer::{TokenizerCommand, run_tokenizer};
use std::fs;
use std::path::PathBuf;

use args::Cli;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing subscriber if --tracing flag is passed
    // INFO+ goes to stdout, ALL levels go to qwen3-tts.log
    if cli.tracing {
        use tracing_subscriber::{
            EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt,
        };

        // Create log file in debug/ directory with timestamp
        let debug_dir = std::path::Path::new("debug");
        if !debug_dir.exists() {
            std::fs::create_dir_all(debug_dir).expect("Failed to create debug directory");
        }
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let log_path = debug_dir.join(format!("qwen3-tts_{}.log", timestamp));
        let log_file = std::fs::File::create(&log_path).expect("Failed to create log file");

        // File layer: ALL levels (trace and above)
        let file_layer = fmt::layer()
            .with_writer(std::sync::Mutex::new(log_file))
            .with_ansi(false)
            .with_filter(EnvFilter::new("trace"));

        tracing_subscriber::registry().with(file_layer).init();

        println!("\n\tDebug logs written to {}\n", log_path.display());
    }

    // Convert CLI args to lib structs
    let io_args = cli.to_io_args();
    let model_args = cli.to_model_args();
    let mut gen_args = cli.to_generation_args();
    let voice_args = cli.to_voice_args();
    let synthesis_mode = cli.to_synthesis_mode();

    // Set random seed if specified (for reproducible generation)
    if let Some(seed) = gen_args.seed {
        set_seed(seed);
        tracing::debug!(seed, "Random seed set");
    }

    println!("Qwen3-TTS CLI");
    println!();

    // Parse device
    let device = match model_args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" | "cuda:0" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("CUDA support not compiled. Rebuild with --features cuda")
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)?
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("Metal support not compiled. Rebuild with --features metal")
            }
        }
        other => bail!("Unknown device: {}. Use cpu, cuda, or metal", other),
    };

    // Parse dtype
    let mut dtype = match model_args.dtype.as_str() {
        "f32" | "float32" => DType::F32,
        "f16" | "float16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        other => bail!("Unknown dtype: {}. Use f32, f16, or bf16", other),
    };

    // CPU doesn't support BF16 or F16 matmul in Candle, fall back to F32
    if matches!(device, Device::Cpu) && matches!(dtype, DType::BF16 | DType::F16) {
        tracing::warn!(
            requested_dtype = %model_args.dtype,
            "CPU does not support requested dtype matmul, using F32 instead"
        );
        dtype = DType::F32;
    }

    // Handle tokenizer subcommand separately
    if let Some(args::CliSynthesisMode::Tokenizer { command }) = &cli.mode {
        let lib_command = match command {
            args::TokenizerCommand::Encode { input, output } => TokenizerCommand::Encode {
                input: input.clone(),
                output: output.clone(),
            },
            args::TokenizerCommand::Decode { input, output } => TokenizerCommand::Decode {
                input: input.clone(),
                output: output.clone(),
            },
            args::TokenizerCommand::Roundtrip {
                input,
                output,
                save_codes,
            } => TokenizerCommand::Roundtrip {
                input: input.clone(),
                output: output.clone(),
                save_codes: save_codes.clone(),
            },
        };
        return run_tokenizer(&lib_command, &model_args, &device, dtype);
    }

    // Determine batch items to process
    let batch_items = load_batch_items(&io_args, &voice_args)?;
    let is_batch_mode = batch_items.len() > 1;

    // Create output directory for batch mode if needed
    if is_batch_mode {
        let default_dir = PathBuf::from(".");
        let output_dir = io_args.output_dir.as_ref().unwrap_or(&default_dir);
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).context("Failed to create output directory")?;
        }
    }

    // Determine synthesis mode from args
    let mode = determine_mode(&io_args, &voice_args, synthesis_mode.as_ref())?;

    // Determine which model to use
    let model_path = get_model_path(&model_args, &mode)?;

    tracing::info!(
        model = %model_path.display(),
        device = ?device,
        dtype = ?dtype,
        "Model configuration"
    );
    if is_batch_mode {
        tracing::info!(
            batch_items = batch_items.len(),
            output_dir = ?io_args.output_dir,
            "Batch mode"
        );
    } else {
        tracing::info!(
            text = ?batch_items[0].text,
            output = ?io_args.output,
            language = %voice_args.language,
            "Single item mode"
        );
    }

    // Log generation parameters
    if gen_args.greedy {
        tracing::info!("Sampling: greedy (deterministic)");
    } else {
        tracing::info!(
            temperature = gen_args.temperature.unwrap_or(0.9),
            top_k = gen_args.top_k.unwrap_or(50),
            top_p = gen_args.top_p.unwrap_or(1.0),
            repetition_penalty = ?gen_args.repetition_penalty,
            "Sampling parameters"
        );
        // Log subtalker params if any are customized
        if gen_args.subtalker_temperature.is_some()
            || gen_args.subtalker_top_k.is_some()
            || gen_args.subtalker_top_p.is_some()
            || gen_args.no_subtalker_sample
        {
            tracing::info!(
                temperature = gen_args.subtalker_temperature.unwrap_or(0.9),
                top_k = gen_args.subtalker_top_k.unwrap_or(50),
                top_p = gen_args.subtalker_top_p.unwrap_or(1.0),
                do_sample = !gen_args.no_subtalker_sample,
                "Subtalker parameters"
            );
        }
    }

    // Load the model
    println!("Loading model...");
    let loader_config = LoaderConfig {
        dtype,
        load_tokenizer: true,
        load_text_tokenizer: true,
        load_generate_config: true,
        use_flash_attn: cli.flash_attn,
    };

    let loader = ModelLoader::from_local_dir(&model_path)
        .map_err(|e| anyhow::anyhow!("Failed to create model loader: {}", e))?;

    let mut model = loader
        .load_tts_model(&device, &loader_config)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    println!("Model loaded!");
    tracing::info!(
        model_type = ?model.model_type(),
        has_text_tokenizer = model.has_text_processor(),
        "Model details"
    );

    // Debug output: model configuration
    if io_args.debug {
        let config = model.generate_defaults();
        tracing::debug!(
            max_new_tokens = ?config.max_new_tokens,
            temperature = ?config.temperature,
            top_k = ?config.top_k,
            top_p = ?config.top_p,
            repetition_penalty = ?config.repetition_penalty,
            do_sample = ?config.do_sample,
            "Model generation config"
        );
    }

    // Estimate max tokens from text length if requested
    if gen_args.estimate_max_tokens {
        let rate = model.tokenizer_rate();
        let mut max_estimated = 0usize;
        for item in &batch_items {
            let token_count = model
                .tokenize_text(&item.text)
                .map(|t| t.len())
                .unwrap_or(item.text.len());
            let estimated_duration = token_count as f64 / 3.0;
            let estimated = (estimated_duration * rate as f64 * 1.2) as usize;
            max_estimated = max_estimated.max(estimated);
        }
        let estimated = max_estimated.max(256);
        println!(
            "Estimated max-tokens: {} (tokenizer rate: {}Hz)",
            estimated, rate
        );
        gen_args.max_tokens = estimated;
    }

    // Perform synthesis based on mode
    match mode {
        DetectedMode::CustomVoice { speaker, instruct } => {
            tracing::info!(
                mode = "CustomVoice",
                speaker = %speaker,
                instruct = ?instruct,
                "Synthesis mode"
            );

            // Process batch items
            for (i, item) in batch_items.iter().enumerate() {
                let output_path =
                    get_output_path(&io_args, i, item.output.as_deref(), is_batch_mode);
                let language = item.language.as_deref().unwrap_or(&voice_args.language);

                if is_batch_mode {
                    tracing::info!(
                        progress = format!("[{}/{}]", i + 1, batch_items.len()),
                        text = ?item.text,
                        "Processing batch item"
                    );
                }

                let synth_item = qwen3_tts::io::SynthesisItem {
                    text: item.text.clone(),
                    language: language.to_string(),
                    output: output_path,
                };

                synthesize_custom_voice_item(
                    &model,
                    &gen_args,
                    &io_args,
                    &speaker,
                    instruct.as_deref(),
                    &synth_item,
                )?;
            }
        }
        DetectedMode::VoiceDesign { description } => {
            tracing::info!(
                mode = "VoiceDesign",
                description = %description,
                "Synthesis mode"
            );

            // Process batch items
            for (i, item) in batch_items.iter().enumerate() {
                let output_path =
                    get_output_path(&io_args, i, item.output.as_deref(), is_batch_mode);
                let language = item.language.as_deref().unwrap_or(&voice_args.language);

                if is_batch_mode {
                    tracing::info!(
                        progress = format!("[{}/{}]", i + 1, batch_items.len()),
                        text = ?item.text,
                        "Processing batch item"
                    );
                }

                let synth_item = qwen3_tts::io::SynthesisItem {
                    text: item.text.clone(),
                    language: language.to_string(),
                    output: output_path,
                };

                synthesize_voice_design_item(
                    &model,
                    &gen_args,
                    &io_args,
                    &description,
                    &synth_item,
                )?;
            }
        }
        DetectedMode::VoiceClone {
            ref_audio,
            ref_text,
            x_vector_only,
        } => {
            tracing::info!(
                mode = "VoiceClone",
                load_prompt = ?io_args.load_prompt,
                ref_audio = ?ref_audio,
                ref_text = ?ref_text,
                x_vector_only = x_vector_only,
                save_prompt = ?io_args.save_prompt,
                "Synthesis mode"
            );

            // Process batch items with voice cloning
            for (i, item) in batch_items.iter().enumerate() {
                let output_path =
                    get_output_path(&io_args, i, item.output.as_deref(), is_batch_mode);
                let language = item.language.as_deref().unwrap_or(&voice_args.language);

                if is_batch_mode {
                    tracing::info!(
                        progress = format!("[{}/{}]", i + 1, batch_items.len()),
                        text = ?item.text,
                        "Processing batch item"
                    );
                }

                let synth_item = qwen3_tts::io::SynthesisItem {
                    text: item.text.clone(),
                    language: language.to_string(),
                    output: output_path,
                };

                let clone_params = qwen3_tts::io::VoiceCloneParams {
                    ref_audio: ref_audio.clone(),
                    ref_text: ref_text.clone(),
                    x_vector_only,
                    save_prompt: i == 0, // Only save on first item
                };

                synthesize_voice_clone_item(
                    &mut model,
                    &gen_args,
                    &io_args,
                    &clone_params,
                    &synth_item,
                )?;

                if i == 0 && let Some(ref save_path) = io_args.save_prompt {
                    println!("Saved voice prompt to {}", save_path.display());
                }
            }
        }
    }

    if is_batch_mode {
        println!("Batch complete: {} items", batch_items.len());
    } else {
        let output_path = get_output_path(&io_args, 0, batch_items[0].output.as_deref(), false);
        println!("File saved to {}", output_path.display());
    }

    // Print timing summary if timing feature is enabled
    #[cfg(feature = "timing")]
    qwen3_tts::nn::timing::print_timings();

    Ok(())
}
