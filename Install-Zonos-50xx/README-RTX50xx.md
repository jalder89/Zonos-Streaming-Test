# Zonos Installation for RTX 50xx Series GPUs

This folder contains installation scripts specifically designed for RTX 50xx series GPUs (RTX 5090, 5080, etc.) which require special compatibility considerations.

## RTX 50xx Requirements

**Important**: RTX 50xx series GPUs currently only support the **transformer** model due to PyTorch compatibility requirements. The hybrid model is not yet supported on these new GPUs.

## Installation Steps

### 1. Run Installation Script
```batch
InstallZonos.bat
```

This will:
- Set up WSL environment with proper CUDA support
- Install PyTorch nightly builds with CUDA 12.8
- Install Zonos with transformer model support
- Install FastAPI and Uvicorn for HerikaServer integration

### 2. Start Zonos for HerikaServer
```batch
StartZonosStreaming.bat
```

This starts the streaming service optimized for HerikaServer integration.

**Alternative**: For standalone Gradio interface:
```batch
StartZonos.bat
```

## Key Differences from Standard Installation

### Model Support
- **RTX 50xx**: Transformer model only
- **Other GPUs**: Both transformer and hybrid models supported

### PyTorch Version
- **RTX 50xx**: PyTorch nightly builds with CUDA 12.8
- **Other GPUs**: Stable PyTorch releases

### Performance
- **Transformer Model**: Faster inference, optimized for RTX 50xx architecture
- **Hybrid Model**: Higher quality audio, requires older GPU architectures

## Configuration for HerikaServer

When using with HerikaServer, ensure your configuration uses transformer model:

```php
$TTS["ZONOS"]["model"]='transformer';  // Required for RTX 50xx
$TTS["ZONOS"]["endpoint"]='http://127.0.0.1:8765';
```

## Troubleshooting RTX 50xx Issues

### CUDA Compatibility
If you encounter CUDA errors:
1. Ensure you have the latest NVIDIA drivers (566.03 or newer)
2. Verify CUDA 12.8 toolkit is installed
3. Check PyTorch nightly builds are properly installed

### Model Loading Errors
If the hybrid model fails to load:
- This is expected on RTX 50xx
- The installation scripts automatically use transformer model
- Verify the service starts with transformer model in logs

### Performance Optimization
For optimal RTX 50xx performance:
- Use smaller chunk schedules: `[8,9,10,12,15]`
- Increase batch sizes if available
- Monitor GPU memory usage

## Files in This Directory

- `InstallZonos.bat` - Main installation script
- `StartZonos.bat` - Start Gradio interface
- `StartZonosStreaming.bat` - Start HerikaServer streaming service
- `zonos_install` - WSL installation script
- `zonos_download_models` - Model download script
- `download_models.py` - Python model downloader
- `start_zonos` - Gradio startup script
- `start_zonos_streaming` - Streaming service startup script

## Support

For RTX 50xx specific issues:
1. Verify your GPU is properly detected: `nvidia-smi`
2. Check CUDA version compatibility: `nvcc --version`
3. Monitor GPU memory during model loading
4. Ensure transformer model is being used (not hybrid)

For general Zonos integration issues, refer to the main `ZONOS-INTEGRATION.md` documentation.