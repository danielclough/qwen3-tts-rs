# Testing Guide

## Quick Start

```bash
# CPU only
cargo test --features test-all

# With CUDA (Linux/Windows with NVIDIA GPU)
cargo test --features test-all-cuda -- --test-threads=1

# With Metal (macOS)
cargo test --features test-all-macos -- --test-threads=1
```

**Note:** First run downloads ~1-2GB of model weights from HuggingFace.

**Important:** GPU tests must run with `--test-threads=1` to avoid out-of-memory errors from parallel model loading.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `integration-tests` | Enables integration tests (model downloads) |
| `v1-ref-tests` | Enables V1 tokenizer reference tests |
| `onnx-xvector` | Enables ONNX x-vector extraction (V1 tokenizer). Requires system `protoc`. |
| `test-all` | `integration-tests` + `v1-ref-tests` |
| `test-all-cuda` | `test-all` + `cuda` + `audio-loading` |
| `test-all-macos` | `test-all` + `metal` + `audio-loading` |

Note the difference between runtime features and test features:

- **`cuda`** (or `metal`) only enables GPU support at compile time — it links against CUDA/Metal libraries so the code can run on a GPU. It does not enable any tests.
- **`test-all-cuda`** (or `test-all-macos`) is a convenience shorthand that combines GPU support with all the test-enabling features (`integration-tests`, `v1-ref-tests`, `audio-loading`). Use this to run the full test suite.

GPU support requires compile-time features because candle links against CUDA/Metal libraries. There's no runtime auto-detection.

To also test V1 ONNX x-vector extraction, add the `onnx-xvector` feature:

```bash
cargo test --features test-all-cuda,onnx-xvector -- --test-threads=1
```

## Test Categories

### Unit Tests (no features required)

```bash
cargo test -p qwen3_tts
```

Tests struct creation, validation, config parsing, sox_norm, kaldi_fbank. No model downloads.

To include ONNX x-vector unit tests:

```bash
cargo test -p qwen3_tts --features onnx-xvector
```

### Integration Tests (require `integration-tests` feature)

Located in `tests/integration_tests.rs`:

| Module | Tests |
|--------|-------|
| `loader_tests` | Model loading, config parsing |
| `text_processor_tests` | Tokenization, unicode, edge cases |
| `tokenizer_tests` | Audio tokenizer encode/decode |
| `generation_tests` | Speech generation (CustomVoice, VoiceDesign, VoiceClone) |
| `error_path_tests` | Invalid inputs, wrong model types |

### Slow Tests

Some tests are marked `#[ignore]` because they require full model loading with audio tokenizer:

```bash
# Run ignored tests
cargo test --features test-all-cuda -- --ignored

# Run all tests including ignored
cargo test --features test-all-cuda -- --test-threads=1 --include-ignored
```

## Developer Checks

Run these before submitting changes to catch compile errors and lint issues early.

### cargo check

Faster than a full build — validates that the code compiles without producing binaries.

```bash
# Default features (CPU)
cargo check -p qwen3_tts

# With all optional features
cargo check -p qwen3_tts --features onnx-xvector,audio-loading

# CUDA
cargo check -p qwen3_tts --features cuda

# CLI crate
cargo check -p qwen3_tts_cli
```

### clippy

```bash
# Default
cargo clippy -p qwen3_tts

# With optional features
cargo clippy -p qwen3_tts --features onnx-xvector,audio-loading

# Entire workspace
cargo clippy --workspace
```

### Full pre-commit checklist

```bash
cargo check --release  --all-targets --features cuda,flash-attn,test-all-cuda,onnx-xvector,audio-loading
cargo test --release --features cuda,flash-attn,test-all-cuda,onnx-xvector,audio-loading -- --test-threads=1
cargo clippy --release --workspace  --features cuda,flash-attn,test-all-cuda,onnx-xvector,audio-loading
```

## Troubleshooting

**CUDA_ERROR_OUT_OF_MEMORY:**
Tests run in parallel by default. Use `--test-threads=1` to run sequentially:
```bash
cargo test --features test-all-cuda -- --test-threads=1
```