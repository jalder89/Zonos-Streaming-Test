# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zonos-v0.1 is an advanced open-weight text-to-speech model providing high-quality speech generation with zero-shot voice cloning, multilingual support, and real-time streaming capabilities. Trained on 200k+ hours of multilingual speech data.

## Development Commands

### Setup
```bash
# System dependencies
brew install espeak-ng  # macOS
apt install -y espeak-ng  # Ubuntu

# Python dependencies  
uv sync                    # Basic install
uv sync --extra compile   # Hybrid model support (requires RTX 3000+)
```

### Running Applications
```bash
uv run gradio_interface.py    # Web interface (recommended)
uv run sample.py             # Basic CLI sample
uv run streaming_sample.py   # Streaming CLI sample
```

### Docker
```bash
docker compose up           # Production deployment
docker build -t zonos .     # Development container
```

### Linting
Uses Ruff with 120-character line length (configured in pyproject.toml)

## Architecture

### Core Components

1. **Model Pipeline**: `zonos/model.py`
   - DAC-based autoencoder for 44kHz audio output
   - Transformer or hybrid (Mamba SSM) backbone architectures
   - Text → eSpeak phonemes → backbone → DAC tokens → audio

2. **Conditioning System**: `zonos/conditioning.py`
   - Modular framework for emotion, pitch, speaking rate, quality controls
   - Language support: English, Japanese, Chinese, French, German

3. **Sampling**: `zonos/sampling.py`
   - Chunked generation for real-time streaming
   - Advanced sampling strategies (CFG, min-p)
   - Configurable chunk schedules for low latency

4. **Speaker Cloning**: `zonos/speaker_cloning.py`
   - LDA-based voice embedding from 10-30s reference audio
   - Zero-shot cloning without fine-tuning

### Entry Points

- **`gradio_interface.py`**: Full-featured web UI with streaming support
- **`sample.py`**: Basic text-to-speech with speaker cloning
- **`streaming_sample.py`**: Real-time audio streaming demonstration

### Model Variants

- `Zyphra/Zonos-v0.1-transformer`: Standard model (broad GPU compatibility)
- `Zyphra/Zonos-v0.1-hybrid`: Advanced Mamba SSM model (RTX 3000+ required)

## Hardware Requirements

- **GPU**: 6GB+ VRAM recommended (RTX 3000+ for hybrid models)
- **CPU**: Supported but significantly slower
- **Performance**: ~2x real-time factor on RTX 4090

## Key Features

- Zero-shot voice cloning from reference audio
- Real-time streaming with chunked generation
- Multilingual text processing via eSpeak
- Fine-grained prosody and emotion control
- Audio prefix continuation
- Docker deployment ready