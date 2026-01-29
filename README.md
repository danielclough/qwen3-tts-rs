# Qwen3-TTS-RS

A Rust implementation of the Qwen3-TTS text-to-speech model using the [Candle](https://github.com/huggingface/candle) ML framework.

![qwen3-tts-rs logo](./qwen3-tts-rs.png)

## Features

- Complete implementation of the Qwen3-TTS architecture
- Speaker encoder (ECAPA-TDNN) for voice cloning
- Audio tokenizer support for both V1 (25Hz) and V2 (12Hz) pipelines
- ONNX-based x-vector speaker embedding extraction (V1 tokenizer, via `onnx-xvector` feature)
- Automatic audio resampling during voice cloning (any input sample rate → 24kHz, via `audio-loading` feature)
- Three synthesis modes:
  - **CustomVoice**: Use predefined speaker voices
  - **VoiceDesign**: Create voices from natural language descriptions
  - **VoiceClone**: Clone voices from reference audio with in-context learning (ICL) support
- Batch processing for multiple texts (TXT or JSON input)
- Voice prompt caching for faster repeated generation (`--save-prompt` / `--load-prompt`)
- URL-based audio loading for voice cloning (via `audio-loading` feature)
- Standalone tokenizer CLI for audio codec testing
- Full control over generation parameters (temperature, top-k, top-p, repetition penalty, seed)
- Multi-language support: Chinese, English, Japanese, Korean, French, German, Spanish (+ auto-detect)

## Installation

### System Dependencies

**Rust** (1.85+):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Linux (Debian/Ubuntu):**

```bash
# Required: build essentials
sudo apt-get -y install build-essential pkg-config libssl-dev

# For onnx-xvector feature (V1 tokenizer speaker embeddings):
sudo apt-get -y install protobuf-compiler
```

**Linux (Fedora/RHEL):**

```bash
sudo dnf install gcc gcc-c++ openssl-devel pkg-config

# For onnx-xvector feature:
sudo dnf install protobuf-compiler
```

**macOS:**

```bash
# Xcode command line tools (if not already installed)
xcode-select --install

# For onnx-xvector feature:
brew install protobuf
```

**CUDA (Linux, for GPU acceleration):**

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (11.8+ recommended). Ensure `nvcc` is on your PATH.

### Building

```bash
# Basic build (CPU only)
cargo build --release

# With audio file loading (WAV, MP3, FLAC, etc.) and resampling
cargo build --release --features audio-loading

# CUDA GPU acceleration
cargo build --release --features cuda

# CUDA with Flash Attention
cargo build --release --features flash-attn

# macOS Metal acceleration
cargo build --release --features metal

# With ONNX x-vector extraction (V1 tokenizer voice cloning)
cargo build --release --features onnx-xvector

# Combine features as needed
cargo build --release --features cuda,flash-attn,audio-loading
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `audio-loading` | Audio file loading (symphonia) and resampling (rubato). Enables loading WAV/MP3/FLAC/OGG files, URL-based voice cloning, and automatic sample rate conversion. |
| `cuda` | NVIDIA CUDA GPU acceleration |
| `cudnn` | cuDNN acceleration (requires `cuda`) |
| `flash-attn` | Flash Attention (requires `cuda`) |
| `metal` | Apple Metal GPU acceleration (macOS) |
| `accelerate` | Apple Accelerate framework (macOS) |
| `mkl` | Intel MKL acceleration |
| `onnx-xvector` | ONNX-based x-vector speaker embedding extraction for V1 tokenizer. Requires system `protoc`. |
| `timing` | Performance timing instrumentation |

## CLI Usage

### CLI Reference

All available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-t, --text` | | Text to synthesize (required unless `--file` is used) |
| `-o, --output` | `output.wav` | Output WAV file path |
| `-f, --file` | | Input file for batch processing (`.json` or `.txt`) |
| `--output-dir` | | Output directory for batch mode |
| `-l, --language` | `auto` | Language code (`auto`, `chinese`, `english`, `japanese`, `korean`, `french`, `german`, `spanish`) |
| `-M, --model` | | HuggingFace model ID (auto-downloads) |
| `-p, --model-path` | | Local model directory (overrides `--model`) |
| `--device` | `cuda` (Linux) / `metal` (macOS) | Compute device (`cpu`, `cuda`, `metal`) |
| `--dtype` | `bf16` | Data type (`f32`, `f16`, `bf16`) |
| **Voice selection** | | |
| `--speaker` | | Speaker name for CustomVoice mode (e.g. `vivian`) |
| `--instruct` | | Instruction for the speaker |
| `--voice-design` | | Voice description for VoiceDesign mode |
| `--ref-audio` | | Path or URL to reference audio for VoiceClone mode |
| `--ref-text` | | Transcript of the reference audio (enables ICL mode) |
| `--x-vector-only` | `false` | Force x-vector only mode (skip ICL) |
| **Prompt caching** | | |
| `--save-prompt` | | Save computed voice prompt to file |
| `--load-prompt` | | Load voice prompt from file (skips ref audio processing) |
| **Generation** | | |
| `--max-tokens` | `2048` | Maximum tokens to generate |
| `--temperature` | `0.9` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling |
| `--top-p` | `1.0` | Nucleus sampling threshold |
| `--repetition-penalty` | | Repetition penalty |
| `--seed` | | Random seed for reproducibility |
| `--greedy` | `false` | Greedy (deterministic) decoding |
| `--subtalker-temperature` | `0.9` | Subtalker (acoustic codebook) temperature |
| `--subtalker-top-k` | `50` | Subtalker top-k |
| `--subtalker-top-p` | `1.0` | Subtalker nucleus sampling |
| `--no-subtalker-sample` | `false` | Greedy subtalker decoding |
| **Debug** | | |
| `--debug` | `false` | Enable debug output |
| `--tracing` | `false` | Enable tracing (writes to `debug/` directory) |
| `--flash-attn` | `false` | Use flash attention (requires `flash-attn` feature) |

### Basic Text-to-Speech

```bash
# Using a predefined speaker (CustomVoice mode)
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --output hello.wav

# With language specification
cargo run --release -- \
    --text "你好，世界！" \
    --speaker vivian \
    --language chinese \
    --output hello_chinese.wav
```

### Synthesis Modes

#### CustomVoice (Predefined Speakers)

Use built-in speaker voices with optional instructions:

```bash
cargo run --release -- \
    --text "Welcome to our service." \
    --speaker vivian \
    --instruct "Speak warmly and professionally" \
    --output welcome.wav
```

#### VoiceDesign (Natural Language Description)

Create a voice from a text description:

```bash
cargo run --release -- \
    --text "Hello, I'm your new assistant." \
    --voice-design "A warm, friendly female voice with a slight British accent" \
    --output designed_voice.wav
```

#### VoiceClone (Reference Audio)

Clone a voice from reference audio. There are two tokenizer versions, and the correct one is selected automatically based on the model you load:

- **V2 (12Hz)** models use a built-in ECAPA-TDNN speaker encoder. No extra features needed.
- **V1 (25Hz)** models use an external ONNX speaker encoder. Requires building with `--features onnx-xvector`.

Both support two cloning modes:

- **X-vector only** (omit `--ref-text`): Uses only the speaker embedding. Faster, works well for general voice matching.
- **ICL mode** (provide `--ref-text`): Uses both the speaker embedding and encoded audio codes for in-context learning. Higher quality voice cloning.

Reference audio at any sample rate is automatically resampled (requires `audio-loading` feature).

```bash
# V2 (12Hz) — x-vector only mode (no --ref-text)
cargo run --release -- \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio reference.wav \
    --text "Quick voice cloning." \
    --output cloned_fast.wav

# V2 (12Hz) — ICL mode (with --ref-text)
cargo run --release -- \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio reference.wav \
    --ref-text "The transcript of the reference audio." \
    --text "This is my cloned voice speaking." \
    --output output_cloned.wav

# V1 (25Hz) — requires onnx-xvector feature (good for longform text)
cargo run --release --features onnx-xvector -- \
    --model Qwen/Qwen3-TTS-25Hz-0.6B-Base \
    --ref-audio reference.wav \
    --ref-text "The transcript of the reference audio." \
    --text "This is my cloned voice speaking." \
    --output output_cloned.wav

# From URL (requires audio-loading feature)
cargo run --release --features cuda,flash-attn,audio-loading -- \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav" \
    --ref-text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." \
    --text "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." \
    --output output_cloned.wav
```

### Batch Processing

#### Using TXT

Create a text file with one text per line:

```bash
# inputs.txt
Hello, this is the first sentence.
This is the second sentence.
And here's the third one.
```

```bash
cargo run --release -- \
    --file inputs.txt \
    --speaker vivian \
    --output-dir ./outputs/
```

This generates `outputs/output_0.wav`, `outputs/output_1.wav`, etc.

#### Using JSON

For more control, use JSON format (detected automatically from `.json` extension):

```json
{
  "items": [
    {"text": "Hello in English!", "language": "english"},
    {"text": "你好！", "language": "chinese", "output": "chinese_greeting.wav"},
    {"text": "Another English sentence.", "language": "english"}
  ]
}
```

```bash
cargo run --release -- \
    --file inputs.json \
    --speaker vivian \
    --output-dir ./outputs/
```

### Voice Prompt Caching

Save computed voice prompts for reuse (avoids recomputing speaker embeddings):

```bash
# Save voice prompt while generating
cargo run --release -- \
    --text "First generation." \
    --ref-audio reference.wav \
    --ref-text "Reference transcript." \
    --save-prompt voice_prompt.safetensors \
    --output first.wav

# Reuse saved prompt (faster, no need for reference audio)
cargo run --release -- \
    --text "Second generation with same voice." \
    --load-prompt voice_prompt.safetensors \
    --output second.wav

# Create prompt without generating audio
cargo run --release -- \
    --ref-audio reference.wav \
    --ref-text "Reference transcript." \
    --save-prompt voice_prompt.safetensors
```

### Generation Parameters

#### Talker Parameters (Semantic Token Generation)

```bash
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --temperature 0.9 \
    --top-k 50 \
    --top-p 1.0 \
    --repetition-penalty 1.05 \
    --max-tokens 2048 \
    --output output.wav

# Greedy decoding (deterministic)
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --greedy \
    --output output.wav

# Set random seed for reproducibility
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --seed 42 \
    --output output.wav
```

##### Max Tokens (default: 2048)

If you want to generate long form text you will need to adjust the `--max-tokens`.

The "Hz" in the model names literally means "tokens per second".

- v1 25Hz = 25 tokens/second = 40ms per token
- v2 12Hz = 12.5 tokens/second = 80ms per token

Given: `tokens = duration_seconds × token_rate_hz`

| max_tokens | 12Hz(v2)| 25Hz(v1)|
|------------|-------- |---------|
| 2,000      | 2m 40s  | 1m 20s  |
| 4,000      | 5m 20s  | 2m 40s  |
| 8,000      | 10m     | 5m 20s  |
| 16,000     | 21m     | 10m     |
| 32,000     | 42m     | 20m     |

##### Per Page

**Reading time for 12 pages**
- Average: 500 words per page
- Speech rate: 150 words per minute (conversational pace)

| Pages | Words  | Duration | 12Hz    | 25Hz    |
|-------|------- |----------|---------|---------|
| 1     | 500    | 3.3 min  | 2,500   | 5,000   |
| 5     | 2,500  | 17 min   | 12,750  | 25,500  |
| 12    | 6,000  | 40 min   | 30,000  | 60,000  |
| 25    | 12,500 | 83 min   | 62,250  | 124,500 |
```

#### Subtalker Parameters (Acoustic Token Generation)

Control the code predictor that generates codebooks 1-31:

```bash
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --subtalker-temperature 0.9 \
    --subtalker-top-k 50 \
    --subtalker-top-p 1.0 \
    --output output.wav

# Disable subtalker sampling (greedy)
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --no-subtalker-sample \
    --output output.wav
```

### Hardware Options

```bash
# Use CPU
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --device cpu \
    --output output.wav

# Use CUDA GPU
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --device cuda \
    --output output.wav

# Use Metal (macOS)
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --device metal \
    --output output.wav

# Set data type
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --dtype bf16 \
    --output output.wav
```

### Tokenizer CLI

Standalone CLI for audio encoding/decoding (codec testing):

```bash
# Encode audio to codes
cargo run --release --example tokenizer_cli --features audio-loading -- encode \
    --input audio.wav \
    --output codes.json

# Decode codes back to audio
cargo run --release --example tokenizer_cli --features audio-loading -- decode \
    --input codes.json \
    --output reconstructed.wav

# Round-trip test (encode then decode)
cargo run --release --example tokenizer_cli --features audio-loading -- roundtrip \
    --input audio.wav \
    --output reconstructed.wav
```

The codes JSON format:

```json
{
  "sample_rate": 24000,
  "num_codebooks": 32,
  "codes": [[1995, 1642, ...], ...]
}
```