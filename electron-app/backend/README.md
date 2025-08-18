# Whisper Transcribe Backend

Optimized Flask server for Whisper speech-to-text transcription with better startup performance and progress tracking.

## Features

- **Fast Startup**: Model loading with progress tracking and caching detection
- **Detailed Logging**: Timestamps and progress information throughout the loading process
- **Smart Caching**: Detects cached models to provide accurate startup time estimates
- **Model Information**: Detailed endpoint for model capabilities and status
- **Error Handling**: Comprehensive error handling with detailed error messages
- **Memory Optimization**: Uses the lightweight "tiny" model by default for faster performance

## API Endpoints

### `/health` - Server and Model Status
Returns detailed information about server and model loading status.

**Response when loading:**
```json
{
  "status": "loading",
  "model": "loading", 
  "progress": "Checking/downloading model files...",
  "elapsed_seconds": 5.2,
  "model_name": "tiny",
  "model_size": "39 MB"
}
```

**Response when ready:**
```json
{
  "status": "ok",
  "model": "loaded",
  "model_name": "tiny", 
  "model_size": "39 MB",
  "load_time_seconds": 8.3
}
```

### `/model-info` - Model Information
Returns detailed information about available models and current configuration.

```json
{
  "current_model": "tiny",
  "recommended_model": "base",
  "available_models": ["tiny", "base", "small", "medium", "large"],
  "model_details": {
    "tiny": {
      "size_mb": 39,
      "parameters": "39M",
      "description": "Fastest, suitable for real-time on most devices"
    }
  },
  "loaded": true,
  "loading": false,
  "progress": null
}
```

### `/transcribe` - Audio Transcription
Transcribes audio files. Returns 503 if model is still loading.

## Performance Optimizations

### 1. Cached Model Detection
The server checks if the Whisper model is already cached locally before starting, providing better startup time estimates:

```
✓ Model found in cache - startup will be faster
⚠ Model not cached - will download on first load
  Expected download size: 39 MB
```

### 2. Background Loading
Models load in a separate daemon thread while the Flask server starts, allowing the Electron app to connect and poll for status immediately.

### 3. Model Warm-up
After loading, the model performs a test transcription with silent audio to ensure all components are fully initialized.

### 4. Detailed Progress Tracking
Every step of the loading process is logged with timestamps:

```
2024-01-15 10:30:15 - INFO - Starting Whisper model loading (tiny - 39 MB)...
2024-01-15 10:30:16 - INFO - Checking/downloading model files...
2024-01-15 10:30:18 - INFO - Performing model warm-up...
2024-01-15 10:30:19 - INFO - Model warm-up completed in 0.8 seconds
2024-01-15 10:30:19 - INFO - Whisper model 'tiny' loaded successfully in 4.2 seconds
```

### 5. Memory Efficient
Uses the "tiny" model (39MB) by default for optimal performance on most systems while maintaining good accuracy.

## Model Preloading

For even faster startup, you can pre-download models using the preloader script:

```bash
# Preload the default tiny model
python preload_model.py

# Preload a different model
python preload_model.py base
python preload_model.py small
```

This downloads and caches the model files, reducing first-run startup time from 10-30 seconds to 2-5 seconds.

## Configuration

### Model Selection
Change the model in `server.py`:

```python
WHISPER_MODEL = "tiny"  # Change to: base, small, medium, large
```

### Model Characteristics
- **tiny**: 39MB, fastest, good for real-time
- **base**: 74MB, better accuracy, ~2x slower  
- **small**: 244MB, high accuracy, ~5x slower
- **medium**: 769MB, very high accuracy, ~10x slower
- **large**: 1550MB, best accuracy, ~20x slower

### Performance vs Accuracy Trade-off
The tiny model provides a good balance for most use cases:
- Fast enough for real-time transcription
- Small memory footprint
- Acceptable accuracy for most applications
- Quick startup time

## Troubleshooting

### Slow First Startup
If the first startup is slow (>30 seconds), it's likely downloading the model:
1. Check your internet connection
2. Use the preloader script to download in advance
3. Check the logs for download progress

### Memory Issues
If you experience memory issues:
1. Ensure you're using the "tiny" model
2. Check available RAM (tiny model needs ~150MB)
3. Close other applications

### Model Loading Fails
Check the logs for specific error messages:
```bash
tail -f /path/to/your/logs
```

Common issues:
- Insufficient disk space for model cache
- Network connectivity issues during download
- Insufficient RAM for model loading

## Integration with Electron App

The Electron app should poll the `/health` endpoint during startup to show progress:

```javascript
async function waitForServer() {
  while (true) {
    try {
      const response = await fetch('http://127.0.0.1:5001/health');
      const status = await response.json();
      
      if (status.status === 'ok') {
        console.log('Server ready!');
        break;
      } else if (status.status === 'loading') {
        console.log(`Loading: ${status.progress} (${status.elapsed_seconds}s)`);
        updateUI(status.progress, status.elapsed_seconds);
      }
    } catch (error) {
      console.log('Waiting for server...');
    }
    
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}
```

This provides real-time feedback to users about the loading progress instead of a blank screen.