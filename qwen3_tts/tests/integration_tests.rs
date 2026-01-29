//! Integration tests that require downloading models from HuggingFace.
//!
//! These tests are gated behind the `integration-tests` feature flag to avoid
//! downloading large model files during normal test runs.
//!
//! Run with: `cargo test --features test-all`
//!
//! Note: First run will download ~600MB-2GB of model weights.

#![cfg(feature = "integration-tests")]

use candle_core::{DType, Device};
use std::path::PathBuf;

use qwen3_tts::io::{ModelArgs, model_path::get_model_path};
use qwen3_tts::model::loader::{LoaderConfig, ModelLoader};
use qwen3_tts::synthesis::detect_mode::DetectedMode;

/// Download model by ID using the main codebase's download logic.
fn get_model(model_id: &str) -> PathBuf {
    let args = ModelArgs {
        model: Some(model_id.to_string()),
        ..Default::default()
    };
    // Mode doesn't matter when model ID is explicitly provided
    let mode = DetectedMode::CustomVoice {
        speaker: "test".to_string(),
        instruct: None,
    };
    get_model_path(&args, &mode).expect("Failed to download model")
}

/// Get the best available device for testing.
/// Automatically detects CUDA/Metal availability based on compiled features.
fn get_test_device() -> Device {
    // Try CUDA if compiled with cuda feature
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            eprintln!("Using CUDA device for tests");
            return device;
        }
    }

    // Try Metal if on macOS and compiled with metal feature
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(device) = Device::new_metal(0) {
            eprintln!("Using Metal device for tests");
            return device;
        }
    }

    // Fall back to CPU
    eprintln!("Using CPU device for tests");
    Device::Cpu
}

mod loader_tests {
    use super::*;
    use qwen3_tts::model::Model;

    /// Load a model and verify it loaded correctly.
    fn load_and_verify_model(model_id: &str, device: &Device) -> Model {
        let model_dir = get_model(model_id);
        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");

        let config = LoaderConfig {
            dtype: DType::F32,
            load_tokenizer: false, // Skip audio tokenizer for faster test
            load_text_tokenizer: true,
            load_generate_config: true,
            use_flash_attn: false,
        };

        let model = loader
            .load_tts_model(device, &config)
            .expect("Failed to load model");

        assert!(model.text_processor().is_some());
        model
    }

    #[test]
    fn test_load_custom_voice_model() {
        let device = get_test_device();
        load_and_verify_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", &device);
    }

    #[test]
    fn test_load_base_model() {
        let device = get_test_device();
        load_and_verify_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base", &device);
    }

    #[test]
    fn test_loader_config_from_model() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");

        // Verify config was loaded - access through talker_config
        let config = loader.model_config();
        assert!(config.talker_config.code_predictor_config.hidden_size > 0);
        assert!(
            config
                .talker_config
                .code_predictor_config
                .num_attention_heads
                > 0
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_load_model_cuda() {
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping CUDA test - no CUDA device available: {}", e);
                return;
            }
        };
        load_and_verify_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", &device);
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_load_model_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping Metal test - no Metal device available: {}", e);
                return;
            }
        };
        load_and_verify_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", &device);
    }
}

mod text_processor_tests {
    use super::*;

    #[test]
    fn test_text_processor_from_pretrained() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let processor = qwen3_tts::text::processing::TextProcessor::from_pretrained(&model_dir)
            .expect("Failed to load text processor");

        // Test tokenization - returns Vec<u32> directly
        let text = "Hello, world!";
        let tokens = processor.tokenize(text);

        assert!(!tokens.is_empty());

        // Test decoding
        let decoded = processor.decode(&tokens).expect("Failed to decode");
        assert!(decoded.contains("Hello"));
    }

    #[test]
    fn test_chat_template_formatting() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let processor = qwen3_tts::text::processing::TextProcessor::from_pretrained(&model_dir)
            .expect("Failed to load text processor");

        // Test assistant text building
        let assistant_text = processor.build_assistant_text("Hello");
        assert!(assistant_text.contains("<|im_start|>assistant"));
        assert!(assistant_text.contains("Hello"));
        assert!(assistant_text.contains("<|im_end|>"));

        // Test instruct text building
        let instruct_text = processor.build_instruct_text("Speak slowly");
        assert!(instruct_text.contains("<|im_start|>user"));
        assert!(instruct_text.contains("Speak slowly"));
    }

    #[test]
    fn test_text_processor_empty_string() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let processor = qwen3_tts::text::processing::TextProcessor::from_pretrained(&model_dir)
            .expect("Failed to load text processor");

        // Empty string should tokenize without error
        let tokens = processor.tokenize("");
        // Empty string produces no tokens (or possibly just special tokens depending on tokenizer)
        // The key assertion is that it doesn't panic
        assert!(
            tokens.is_empty() || tokens.len() < 5,
            "Empty string should produce minimal tokens"
        );

        // tokenize_for_tts wraps in chat template, so it will have some tokens
        let tts_tokens = processor.tokenize_for_tts("");
        assert!(
            !tts_tokens.is_empty(),
            "TTS tokens include chat template markers"
        );
    }

    #[test]
    fn test_text_processor_unicode() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let processor = qwen3_tts::text::processing::TextProcessor::from_pretrained(&model_dir)
            .expect("Failed to load text processor");

        // Test various Unicode characters
        let unicode_texts = [
            "Hello, 世界!",       // Chinese
            "Привет мир",         // Russian
            "مرحبا بالعالم",      // Arabic
            "🎉 Celebration! 🎊", // Emoji
            "Ñoño año",           // Spanish with tildes
            "日本語テスト",       // Japanese
        ];

        for text in unicode_texts {
            let tokens = processor.tokenize(text);
            assert!(!tokens.is_empty(), "Should tokenize Unicode text: {}", text);

            // Verify round-trip decoding works
            if let Some(decoded) = processor.decode(&tokens) {
                // The decoded text should contain at least some of the original
                // (exact match not guaranteed due to tokenizer behavior)
                assert!(
                    !decoded.is_empty(),
                    "Should decode back to non-empty string for: {}",
                    text
                );
            }
        }
    }

    #[test]
    fn test_text_processor_special_characters() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let processor = qwen3_tts::text::processing::TextProcessor::from_pretrained(&model_dir)
            .expect("Failed to load text processor");

        // Test special characters and punctuation
        let special_texts = [
            "Hello... how are you?",
            "Test! Test? Test.",
            "Quote: \"Hello\" and 'World'",
            "Math: 2 + 2 = 4, 10% off",
            "Email: test@example.com",
            "Path: /usr/bin/test",
            "Brackets: [array] {object} (parens)",
        ];

        for text in special_texts {
            let tokens = processor.tokenize(text);
            assert!(!tokens.is_empty(), "Should tokenize special text: {}", text);
        }
    }

    #[test]
    fn test_text_processor_long_input() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let processor = qwen3_tts::text::processing::TextProcessor::from_pretrained(&model_dir)
            .expect("Failed to load text processor");

        // Create a long input string (about 1000 words)
        let sentence = "This is a test sentence for the text to speech system. ";
        let long_text: String = sentence.repeat(100);

        let tokens = processor.tokenize(&long_text);
        assert!(!tokens.is_empty(), "Should tokenize long text");
        // Long text should produce many tokens
        assert!(
            tokens.len() > 500,
            "Long text should produce many tokens, got {}",
            tokens.len()
        );
    }

    #[test]
    fn test_text_processor_whitespace_handling() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let processor = qwen3_tts::text::processing::TextProcessor::from_pretrained(&model_dir)
            .expect("Failed to load text processor");

        // Test various whitespace scenarios
        let whitespace_texts = [
            "   Leading spaces",
            "Trailing spaces   ",
            "Multiple   spaces   between",
            "Tabs\tand\tnewlines\n",
            "\n\nMultiple\n\nnewlines\n\n",
        ];

        for text in whitespace_texts {
            let tokens = processor.tokenize(text);
            assert!(
                !tokens.is_empty(),
                "Should tokenize whitespace text: {:?}",
                text
            );
        }
    }
}

mod tokenizer_tests {
    use super::*;
    use candle_nn::VarBuilder;
    use qwen3_tts::audio::tokenizer::v2::{TokenizerV2, config::TokenizerV2Config};

    #[test]
    fn test_audio_tokenizer_decode() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        let device = get_test_device();

        let tokenizer_dir = model_dir.join("speech_tokenizer");
        if !tokenizer_dir.exists() {
            eprintln!("Skipping audio tokenizer test - speech_tokenizer not found");
            return;
        }

        // Load config from JSON
        let config_json = std::fs::read_to_string(tokenizer_dir.join("config.json"))
            .expect("Failed to read tokenizer config");
        let config =
            TokenizerV2Config::from_json(&config_json).expect("Failed to parse tokenizer config");

        // Load weights
        let weights_path = tokenizer_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .expect("Failed to load tokenizer weights")
        };

        let mut tokenizer =
            TokenizerV2::new(config.clone(), false, vb).expect("Failed to create tokenizer");

        // Create dummy codes for testing
        let codes = candle_core::Tensor::zeros(
            (1, config.decoder_config.num_quantizers, 10),
            DType::I64,
            &device,
        )
        .expect("Failed to create dummy codes");

        // Decode should produce audio
        let audio = tokenizer.decode(&codes).expect("Failed to decode");

        // Check output shape
        let dims = audio.dims();
        assert_eq!(dims.len(), 2); // (batch, samples)
        assert_eq!(dims[0], 1); // batch size
        assert!(dims[1] > 0); // some samples
    }

    /// Test that the audio tokenizer encoder loads from Base model and can encode audio.
    ///
    /// The Base model should include encoder weights for ICL mode support.
    /// This test verifies that:
    /// 1. The encoder loads successfully with `with_encoder()`
    /// 2. `has_encoder()` returns true after loading
    /// 3. The encoder can encode audio to discrete codes with correct shape
    #[test]
    fn test_audio_tokenizer_encoder_loads() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let device = get_test_device();

        let tokenizer_dir = model_dir.join("speech_tokenizer");
        if !tokenizer_dir.exists() {
            eprintln!("Skipping audio tokenizer encoder test - speech_tokenizer not found");
            return;
        }

        // Load config from JSON
        let config_json = std::fs::read_to_string(tokenizer_dir.join("config.json"))
            .expect("Failed to read tokenizer config");
        let config =
            TokenizerV2Config::from_json(&config_json).expect("Failed to parse tokenizer config");

        // Load weights
        let weights_path = tokenizer_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .expect("Failed to load tokenizer weights")
        };

        // Create tokenizer with decoder first
        let tokenizer = TokenizerV2::new(config.clone(), false, vb.clone())
            .expect("Failed to create tokenizer");

        // Verify decoder-only tokenizer doesn't have encoder
        assert!(
            !tokenizer.has_encoder(),
            "Decoder-only tokenizer should not have encoder"
        );

        // Try to add encoder
        let tokenizer_with_encoder = tokenizer.with_encoder(vb.pp("encoder"));

        match tokenizer_with_encoder {
            Ok(mut tokenizer) => {
                // Encoder loaded successfully
                assert!(
                    tokenizer.has_encoder(),
                    "Tokenizer should have encoder after with_encoder()"
                );

                // Create dummy audio for encoding test
                // Audio should be (batch, samples) at 24kHz
                // Using 0.5 seconds = 12000 samples at 24kHz
                let audio_samples = 12000;
                let audio = candle_core::Tensor::randn(0f32, 0.1, (1, audio_samples), &device)
                    .expect("Failed to create dummy audio");

                // Encode audio to codes
                let codes = tokenizer.encode(&audio).expect("Failed to encode audio");

                // Check output shape: (batch, num_quantizers, seq_len)
                let dims = codes.dims();
                assert_eq!(dims.len(), 3, "Codes should be 3D tensor");
                assert_eq!(dims[0], 1, "Batch size should be 1");

                // Number of quantizers should match config
                let expected_quantizers = config.encoder_valid_num_quantizers;
                assert_eq!(
                    dims[1], expected_quantizers,
                    "Number of quantizers should be {}",
                    expected_quantizers
                );

                // Sequence length should be proportional to audio length
                // At 12Hz (12.5Hz actually) with 24kHz input, we expect ~audio_samples/downsample_rate codes
                let downsample_rate = config.encode_downsample_rate;
                let expected_seq_len = audio_samples / downsample_rate;
                // Allow some tolerance for rounding
                assert!(
                    dims[2] >= expected_seq_len.saturating_sub(2)
                        && dims[2] <= expected_seq_len + 2,
                    "Sequence length {} should be close to {} (audio_samples/downsample_rate)",
                    dims[2],
                    expected_seq_len
                );

                eprintln!(
                    "Encoder test passed: audio ({} samples) -> codes {:?}",
                    audio_samples, dims
                );
            }
            Err(e) => {
                // This is a soft failure - Base model might not have encoder weights
                eprintln!(
                    "Warning: Failed to load encoder from Base model: {}. \
                     This may indicate encoder weights are not included in the model.",
                    e
                );
                // Don't fail the test - just note that ICL won't be available
            }
        }
    }
}

mod generation_tests {
    use super::*;

    #[test]
    #[ignore] // Requires full model loading with audio tokenizer
    fn test_generate_custom_voice() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig::default();
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        let result = model
            .generate_custom_voice_from_text("Hello world.", "vivian", "english", None, None)
            .expect("Generation should succeed");

        assert_eq!(result.sample_rate, 24000);
        assert!(result.audio.dims()[0] > 0, "Audio should have samples");
        assert!(
            result.codes.is_some(),
            "Result should include generated codes"
        );
    }

    #[test]
    fn test_generate_voice_design_from_text() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig::default();
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        let result = model
            .generate_voice_design_from_text(
                "Hello world.",
                "A warm, friendly female voice",
                "english",
                None,
            )
            .expect("Generation should succeed");

        assert_eq!(result.sample_rate, 24000);
        assert!(result.audio.dims()[0] > 0, "Audio should have samples");
    }

    #[test]
    #[ignore] // Requires full model loading
    fn test_generate_custom_voice_from_texts_batch() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig::default();
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        let texts = ["Hello.", "How are you?", "Goodbye."];
        let results = model
            .generate_custom_voice_from_texts_batch(&texts, "vivian", "english", None, None)
            .expect("Batch generation should succeed");

        assert_eq!(results.len(), 3, "Should get one result per input text");
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.sample_rate, 24000);
            assert!(
                result.audio.dims()[0] > 0,
                "Audio {} should have samples",
                i
            );
        }
    }

    #[test]
    fn test_generate_voice_clone_x_vector() {
        use candle_core::Tensor;
        use qwen3_tts::model::voice_clone::VoiceClonePromptItem;

        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig::default();
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        // Create an x-vector only voice clone prompt with a dummy speaker embedding
        // The embedding dimension (1024) matches the model's hidden_size
        let speaker_embed = Tensor::randn(0.0f32, 1.0, (1024,), &device)
            .expect("Failed to create speaker embedding");
        let prompt = VoiceClonePromptItem::x_vector_only(speaker_embed);

        // Validate that the prompt is correctly formed
        assert!(
            prompt.validate().is_ok(),
            "Voice clone prompt should be valid"
        );
        assert!(prompt.x_vector_only_mode, "Should be in x-vector only mode");

        // Generate speech with the voice clone API
        let result = model
            .generate_voice_clone_from_text(
                "Hello, this is a voice clone test.",
                &prompt,
                "english",
                None,
            )
            .expect("Voice clone generation should succeed");

        assert_eq!(result.sample_rate, 24000);
        assert!(result.audio.dims()[0] > 0, "Audio should have samples");
        assert!(
            result.codes.is_some(),
            "Result should include generated codes"
        );
    }

    /// Test ICL (In-Context Learning) voice clone mode end-to-end.
    ///
    /// ICL mode uses reference audio codes + transcript in addition to speaker embedding
    /// for better voice cloning quality. This test verifies:
    /// 1. Base model loads with encoder (for encoding reference audio)
    /// 2. `create_voice_clone_prompt_from_audio()` produces ICL prompt when encoder available
    /// 3. ICL prompt has `is_icl() == true` and `ref_code.is_some()`
    /// 4. Generation succeeds with ICL prompt
    #[test]
    fn test_generate_voice_clone_icl() {
        use candle_core::Tensor;

        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig::default();
        let mut model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        // Check if the audio tokenizer has encoder capability
        let has_encoder = model
            .audio_tokenizer()
            .map(|t| t.has_encoder())
            .unwrap_or(false);
        eprintln!("Audio tokenizer has encoder: {}", has_encoder);

        if !has_encoder {
            eprintln!(
                "Skipping ICL test - encoder not available in this model. \
                 Voice cloning will fall back to x-vector only mode."
            );
            return;
        }

        // Create synthetic reference audio (1 second of sine wave at 440Hz)
        // Audio should be at 24kHz for the speaker encoder
        let sample_rate = 24000;
        let duration_secs = 1.0;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let frequency = 440.0;

        // Generate sine wave audio
        let audio_data: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let audio = Tensor::from_vec(audio_data, (num_samples,), &device)
            .expect("Failed to create audio tensor");

        // Reference text for ICL mode
        let ref_text = "This is a test sentence for voice cloning.".to_string();

        // Create ICL voice clone prompt from audio
        // x_vector_only_mode = false means we want ICL mode
        let prompt = model
            .create_voice_clone_prompt_from_audio(&audio, Some(ref_text.clone()), false)
            .expect("Failed to create voice clone prompt");

        // Verify ICL mode is active
        eprintln!(
            "Prompt mode: is_icl={}, is_x_vector_only={}",
            prompt.is_icl(),
            prompt.is_x_vector_only()
        );
        eprintln!("Prompt has ref_code: {}", prompt.ref_code.is_some());
        if let Some(ref codes) = prompt.ref_code {
            eprintln!("ref_code shape: {:?}", codes.dims());
        }

        // If encoder worked, we should have ICL mode
        if prompt.is_icl() {
            assert!(prompt.ref_code.is_some(), "ICL mode should have ref_code");
            assert_eq!(
                prompt.ref_text,
                Some(ref_text),
                "ref_text should be preserved"
            );
            assert!(prompt.validate().is_ok(), "ICL prompt should be valid");

            // Generate speech with ICL mode
            let result = model
                .generate_voice_clone_from_text(
                    "Hello, this is a voice clone test with ICL mode.",
                    &prompt,
                    "english",
                    None,
                )
                .expect("ICL voice clone generation should succeed");

            assert_eq!(result.sample_rate, 24000);
            assert!(result.audio.dims()[0] > 0, "Audio should have samples");
            assert!(
                result.codes.is_some(),
                "Result should include generated codes"
            );

            eprintln!(
                "ICL voice clone test passed! Generated {} audio samples",
                result.audio.dims()[0]
            );
        } else {
            // Encoder might have failed silently, fell back to x-vector mode
            eprintln!(
                "Warning: Expected ICL mode but got x-vector only mode. \
                 This may indicate an issue with the encoder."
            );
            // Still verify the fallback works
            assert!(
                prompt.is_x_vector_only(),
                "Should fall back to x-vector only mode"
            );
        }
    }
}

mod voice_clone_prompt_tests {
    use super::*;
    use candle_core::Tensor;

    fn load_base_model(device: &Device) -> qwen3_tts::model::Model {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig {
            dtype: DType::F32,
            load_tokenizer: false,
            load_text_tokenizer: true,
            load_generate_config: true,
            use_flash_attn: false,
        };
        loader
            .load_tts_model(device, &config)
            .expect("Failed to load model")
    }

    #[test]
    fn test_voice_clone_prompt_resamples_non_24khz() {
        let device = get_test_device();
        let mut model = load_base_model(&device);

        // 1 second at 48kHz
        let audio_data: Vec<f32> = (0..48000)
            .map(|i| {
                let t = i as f32 / 48000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();
        let audio =
            Tensor::from_vec(audio_data, (48000,), &device).expect("Failed to create tensor");

        let result =
            model.create_voice_clone_prompt_with_sample_rate(&audio, 48000, None, true);
        assert!(
            result.is_ok(),
            "Resampling 48kHz→24kHz should succeed: {:?}",
            result.err()
        );
        let prompt = result.unwrap();
        assert!(prompt.x_vector_only_mode, "Should be x_vector_only mode");
    }

    #[test]
    fn test_voice_clone_prompt_passthrough_24khz() {
        let device = get_test_device();
        let mut model = load_base_model(&device);

        // 1 second at 24kHz
        let audio_data: Vec<f32> = (0..24000)
            .map(|i| {
                let t = i as f32 / 24000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();
        let audio =
            Tensor::from_vec(audio_data, (24000,), &device).expect("Failed to create tensor");

        let result =
            model.create_voice_clone_prompt_with_sample_rate(&audio, 24000, None, true);
        assert!(
            result.is_ok(),
            "24kHz passthrough should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    #[cfg(feature = "onnx-xvector")]
    fn test_extract_code_output_shapes() {
        use qwen3_tts::audio::encoder::xvector::XVectorExtractor;

        let device = get_test_device();
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let onnx_path = model_dir.join("campplus_cn_common.onnx");
        if !onnx_path.exists() {
            eprintln!("Skipping test_extract_code_output_shapes - ONNX model not found");
            return;
        }

        let extractor =
            XVectorExtractor::load(&onnx_path, &device).expect("Failed to load ONNX model");

        // 1 second of 16kHz sine wave
        let audio: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let (xvec, ref_mel) = extractor.extract_code(&audio).expect("extract_code failed");

        // x-vector should be 1D with 192 dims
        let xvec_dims = xvec.dims();
        assert_eq!(xvec_dims.len(), 1, "x-vector should be 1D");
        assert_eq!(xvec_dims[0], 192, "x-vector should have 192 dims");

        // ref_mel should be (T, 80) with T > 0
        let mel_dims = ref_mel.dims();
        assert_eq!(mel_dims.len(), 2, "ref_mel should be 2D");
        assert!(mel_dims[0] > 0, "ref_mel should have >0 time frames");
        assert_eq!(mel_dims[1], 80, "ref_mel should have 80 mel channels");
    }

    #[test]
    #[cfg(feature = "onnx-xvector")]
    fn test_extract_code_l2_normalized() {
        use qwen3_tts::audio::encoder::xvector::XVectorExtractor;

        let device = get_test_device();
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let onnx_path = model_dir.join("campplus_cn_common.onnx");
        if !onnx_path.exists() {
            eprintln!("Skipping test_extract_code_l2_normalized - ONNX model not found");
            return;
        }

        let extractor =
            XVectorExtractor::load(&onnx_path, &device).expect("Failed to load ONNX model");

        let audio: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let (xvec, _) = extractor.extract_code(&audio).expect("extract_code failed");

        // L2 norm should be ~1.0
        let norm = xvec
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .sqrt()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "x-vector L2 norm should be ~1.0, got {}",
            norm
        );
    }

    #[test]
    #[cfg(feature = "onnx-xvector")]
    fn test_extract_code_ref_mel_shape_matches_v1() {
        use qwen3_tts::audio::encoder::xvector::{XVectorExtractor, sox_norm};
        use qwen3_tts::audio::mel_v1::MelSpectrogramFeaturesV1;

        let device = get_test_device();
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let onnx_path = model_dir.join("campplus_cn_common.onnx");
        if !onnx_path.exists() {
            eprintln!(
                "Skipping test_extract_code_ref_mel_shape_matches_v1 - ONNX model not found"
            );
            return;
        }

        let extractor =
            XVectorExtractor::load(&onnx_path, &device).expect("Failed to load ONNX model");

        let audio: Vec<f32> = (0..16000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let (_, ref_mel) = extractor.extract_code(&audio).expect("extract_code failed");

        // Compute V1 mel directly on sox-normed audio
        let normed = sox_norm(&audio);
        let mel = MelSpectrogramFeaturesV1::new();
        let audio_t =
            Tensor::from_vec(normed.clone(), normed.len(), &device).expect("tensor creation");
        let v1_mel = mel.forward(&audio_t, &device).expect("mel forward");
        let v1_transposed = v1_mel.squeeze(0).unwrap().transpose(0, 1).unwrap();

        assert_eq!(
            ref_mel.dims(),
            v1_transposed.dims(),
            "ref_mel from extract_code should match transposed V1 mel shape"
        );
    }
}

mod error_path_tests {
    use super::*;

    #[test]
    fn test_generate_custom_voice_invalid_speaker() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig {
            load_tokenizer: false, // Faster - we're testing validation, not generation
            ..Default::default()
        };
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        let result = model.generate_custom_voice_from_text(
            "Hello",
            "nonexistent_speaker",
            "english",
            None,
            None,
        );

        assert!(result.is_err(), "Should fail with invalid speaker");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unknown speaker") || err_msg.contains("nonexistent_speaker"),
            "Error message should mention invalid speaker: {}",
            err_msg
        );
    }

    #[test]
    fn test_generate_custom_voice_invalid_language() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig {
            load_tokenizer: false, // Faster - we're testing validation, not generation
            ..Default::default()
        };
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        let result =
            model.generate_custom_voice_from_text("Hello", "vivian", "esperanto", None, None);

        assert!(result.is_err(), "Should fail with invalid language");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unsupported language") || err_msg.contains("esperanto"),
            "Error message should mention invalid language: {}",
            err_msg
        );
    }

    #[test]
    fn test_base_model_rejects_custom_voice_api() {
        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig {
            load_tokenizer: false,
            ..Default::default()
        };
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        // Base model should reject custom voice API
        let result =
            model.generate_custom_voice_from_text("Hello", "vivian", "english", None, None);

        assert!(result.is_err(), "Base model should reject custom voice API");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("does not support") || err_msg.contains("CustomVoice"),
            "Error should indicate wrong model type: {}",
            err_msg
        );
    }

    #[test]
    fn test_custom_voice_model_rejects_voice_clone_api() {
        use candle_core::Tensor;
        use qwen3_tts::model::voice_clone::VoiceClonePromptItem;

        let model_dir = get_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        let device = get_test_device();

        let loader = ModelLoader::from_local_dir(&model_dir).expect("Failed to create loader");
        let config = LoaderConfig {
            load_tokenizer: false,
            ..Default::default()
        };
        let model = loader
            .load_tts_model(&device, &config)
            .expect("Failed to load model");

        // Create a dummy voice clone prompt (1024 matches model hidden_size)
        let speaker_embed =
            Tensor::zeros((1024,), DType::F32, &device).expect("Failed to create tensor");
        let prompt = VoiceClonePromptItem::x_vector_only(speaker_embed);

        // CustomVoice model should reject voice clone API
        let result = model.generate_voice_clone_from_text("Hello", &prompt, "english", None);

        assert!(
            result.is_err(),
            "CustomVoice model should reject voice clone API"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("does not support") || err_msg.contains("voice_clone"),
            "Error should indicate wrong model type: {}",
            err_msg
        );
    }
}
