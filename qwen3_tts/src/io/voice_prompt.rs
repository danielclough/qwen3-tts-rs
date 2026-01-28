#[cfg(feature = "audio-loading")]
use crate::model::voice_clone::VoiceClonePromptItem;
use anyhow::{Context, Result, bail};
use candle_core::DType;
#[cfg(feature = "audio-loading")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "audio-loading")]
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "audio-loading")]
const VOICE_PROMPT_VERSION: u32 = 1;

/// Serialized voice clone prompt for save/load
#[cfg(feature = "audio-loading")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoicePromptFile {
    /// File format version
    version: u32,
    /// Whether x-vector only mode was used
    x_vector_only: bool,
    /// Reference text (if provided)
    ref_text: Option<String>,
    /// Serialized prompt data (base64-encoded safetensors)
    prompt_data: String,
}

/// Save a voice clone prompt to a file
#[cfg(feature = "audio-loading")]
pub fn save_voice_prompt(
    prompt: &VoiceClonePromptItem,
    path: &PathBuf,
    x_vector_only: bool,
    ref_text: Option<&str>,
) -> Result<()> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
    use safetensors::serialize;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap;

    // Convert tensors to TensorViews for serialization
    // The embedding may be in BF16 format, so convert to F32 for serialization
    let spk_data = prompt
        .ref_spk_embedding
        .to_dtype(candle_core::DType::F32)?
        .to_vec1::<f32>()?;
    let spk_shape: Vec<usize> = prompt.ref_spk_embedding.dims().to_vec();

    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    views.insert(
        "ref_spk_embedding".to_string(),
        TensorView::new(
            safetensors::Dtype::F32,
            spk_shape,
            bytemuck::cast_slice(&spk_data),
        )?,
    );

    // Handle optional ref_code
    let code_data: Option<Vec<i64>>;
    let code_shape: Option<Vec<usize>>;
    if let Some(ref code) = prompt.ref_code {
        let code_i64 = code.to_dtype(candle_core::DType::I64)?;
        code_data = Some(code_i64.to_vec2::<i64>()?.into_iter().flatten().collect());
        code_shape = Some(code.dims().to_vec());
    } else {
        code_data = None;
        code_shape = None;
    }

    if let (Some(data), Some(shape)) = (&code_data, &code_shape) {
        views.insert(
            "ref_code".to_string(),
            TensorView::new(
                safetensors::Dtype::I64,
                shape.clone(),
                bytemuck::cast_slice(data),
            )?,
        );
    }

    // Serialize to bytes
    let buffer = serialize(&views, None)?;

    // Encode as base64 and wrap in JSON
    let prompt_file = VoicePromptFile {
        version: VOICE_PROMPT_VERSION,
        x_vector_only,
        ref_text: ref_text.map(|s| s.to_string()),
        prompt_data: BASE64.encode(&buffer),
    };

    let json = serde_json::to_string_pretty(&prompt_file)?;
    fs::write(path, json)?;

    Ok(())
}

/// Load a voice clone prompt from a file
#[cfg(feature = "audio-loading")]
pub fn load_voice_prompt(
    path: &PathBuf,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<VoiceClonePromptItem> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
    use candle_core::safetensors::load_buffer;

    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read voice prompt file: {:?}", path))?;

    let prompt_file: VoicePromptFile = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse voice prompt file: {:?}", path))?;

    if prompt_file.version != VOICE_PROMPT_VERSION {
        bail!(
            "Voice prompt file version mismatch: expected {}, got {}",
            VOICE_PROMPT_VERSION,
            prompt_file.version
        );
    }

    // Decode base64 and load tensors
    let buffer = BASE64
        .decode(&prompt_file.prompt_data)
        .context("Failed to decode voice prompt data")?;

    let tensors = load_buffer(&buffer, device)?;

    // Load embedding and convert to model's dtype (saved as F32 for portability)
    let ref_spk_embedding = tensors
        .get("ref_spk_embedding")
        .context("Voice prompt file missing ref_spk_embedding")?
        .to_dtype(dtype)?;

    let ref_code = tensors
        .get("ref_code")
        .map(|t| t.to_dtype(candle_core::DType::U32))
        .transpose()?;

    // Determine icl_mode: true if we have ref_code and ref_text and not x_vector_only
    let icl_mode =
        ref_code.is_some() && prompt_file.ref_text.is_some() && !prompt_file.x_vector_only;

    Ok(VoiceClonePromptItem {
        ref_code,
        ref_spk_embedding,
        x_vector_only_mode: prompt_file.x_vector_only,
        icl_mode,
        ref_text: prompt_file.ref_text,
    })
}

pub fn write_wav(path: &PathBuf, audio: &candle_core::Tensor, sample_rate: usize) -> Result<()> {
    use hound::{SampleFormat, WavSpec, WavWriter};

    // Flatten the audio tensor to 1D
    // Audio may come as (batch, samples) or (samples,) depending on the model output
    let audio_flat = audio.flatten_all()?;
    let num_samples = audio_flat.dim(0)?;

    // Check for empty audio
    if num_samples == 0 {
        bail!(
            "No audio was generated. The model produced 0 samples. \
             This may indicate an issue with the model weights, configuration, \
             or that the input text is too short/empty."
        );
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec).context("Failed to create WAV file")?;

    // Convert float samples to i16
    let samples = audio_flat.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for sample in samples {
        // Clamp and convert to i16
        let sample = sample.clamp(-1.0, 1.0);
        let sample_i16 = (sample * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}
