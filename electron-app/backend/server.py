#!/usr/bin/env python3

from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import numpy as np
from scipy import signal
import wave
import subprocess
import logging
from datetime import datetime
import time
import sys
import shutil
try:
    from model_config import WHISPER_MODELS, get_recommended_model
except ImportError:
    # Fallback if model_config is not available
    WHISPER_MODELS = {}
    def get_recommended_model():
        return "tiny"

app = Flask(__name__)
# Enable CORS for all routes and origins with all methods
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*"}})

# Set up logging with timestamps
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Disable strict slashes
app.url_map.strict_slashes = False

# Global model variable
model = None
model_loading = False
model_loaded = False
model_loading_progress = "Starting..."
model_loading_start_time = None

# Model size information
MODEL_SIZES = {
    "tiny": "39 MB",
    "base": "74 MB",
    "small": "244 MB",
    "medium": "769 MB",
    "large": "1550 MB"
}

# Use tiny model for fast loading (can be configured)
WHISPER_MODEL = "tiny"

def load_model_async():
    global model, model_loading, model_loaded, model_loading_progress, model_loading_start_time
    model_loading = True
    model_loading_start_time = time.time()
    
    try:
        # Step 1: Log startup
        model_loading_progress = f"Starting Whisper model loading ({WHISPER_MODEL} - {MODEL_SIZES.get(WHISPER_MODEL, 'Unknown size')})..."
        logging.info(model_loading_progress)
        print(f"\n{'='*60}")
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        print(f"Model size: {MODEL_SIZES.get(WHISPER_MODEL, 'Unknown')}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Step 2: Import and download model if needed
        model_loading_progress = "Checking/downloading model files..."
        logging.info(model_loading_progress)
        start_download = time.time()
        
        # This will download the model if not cached
        model = whisper.load_model(WHISPER_MODEL)
        
        download_time = time.time() - start_download
        
        # Step 3: Model loaded
        model_loaded = True
        model_loading = False
        
        total_time = time.time() - model_loading_start_time
        model_loading_progress = f"Model loaded successfully in {total_time:.1f} seconds"
        
        print(f"\n{'='*60}")
        print(f"✓ Model loaded successfully!")
        print(f"  - Download/load time: {download_time:.1f}s")
        print(f"  - Total time: {total_time:.1f}s")
        print(f"  - Model ready for transcription")
        print(f"{'='*60}\n")
        
        logging.info(f"Whisper model '{WHISPER_MODEL}' loaded successfully in {total_time:.1f} seconds")
        
        # Step 4: Warm up the model with a test transcription
        try:
            model_loading_progress = "Warming up model with test transcription..."
            logging.info("Performing model warm-up...")
            warmup_start = time.time()
            
            # Create a small silent audio for warm-up
            import numpy as np
            silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            warmup_result = model.transcribe(silence, language="en", fp16=False)
            
            warmup_time = time.time() - warmup_start
            logging.info(f"Model warm-up completed in {warmup_time:.1f} seconds")
            
            model_loading_progress = f"Model fully initialized and ready!"
            
        except Exception as warmup_error:
            logging.warning(f"Model warm-up failed (non-critical): {warmup_error}")
            # Don't fail the whole loading process for warm-up errors
        
    except Exception as e:
        model_loading = False
        model_loaded = False
        error_time = time.time() - model_loading_start_time if model_loading_start_time else 0
        model_loading_progress = f"Failed to load model after {error_time:.1f}s: {str(e)}"
        
        print(f"\n{'='*60}")
        print(f"✗ Model loading failed!")
        print(f"  - Error: {str(e)}")
        print(f"  - Time elapsed: {error_time:.1f}s")
        print(f"{'='*60}\n")
        
        logging.error(f"Failed to load Whisper model: {e}")
        logging.exception("Full traceback:")

@app.route('/health', methods=['GET'])
def health():
    if model_loaded:
        load_time = time.time() - model_loading_start_time if model_loading_start_time else 0
        return jsonify({
            "status": "ok", 
            "model": "loaded",
            "model_name": WHISPER_MODEL,
            "model_size": MODEL_SIZES.get(WHISPER_MODEL, "Unknown"),
            "load_time_seconds": round(load_time, 1)
        })
    elif model_loading:
        elapsed_time = time.time() - model_loading_start_time if model_loading_start_time else 0
        return jsonify({
            "status": "loading", 
            "model": "loading",
            "progress": model_loading_progress,
            "elapsed_seconds": round(elapsed_time, 1),
            "model_name": WHISPER_MODEL,
            "model_size": MODEL_SIZES.get(WHISPER_MODEL, "Unknown")
        }), 503
    else:
        return jsonify({
            "status": "error", 
            "model": "not_loaded",
            "error": model_loading_progress if "Failed" in model_loading_progress else "Model not loaded"
        }), 503

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about available models and current model status"""
    # Use detailed model info if available
    if WHISPER_MODELS:
        available_models = WHISPER_MODELS
    else:
        # Fallback to simple size info
        available_models = {name: {"size_mb": int(size.split()[0])} for name, size in MODEL_SIZES.items()}
    
    return jsonify({
        "current_model": WHISPER_MODEL,
        "recommended_model": get_recommended_model(),
        "available_models": list(MODEL_SIZES.keys()),
        "model_details": available_models,
        "model_sizes": MODEL_SIZES,
        "loaded": model_loaded,
        "loading": model_loading,
        "progress": model_loading_progress if model_loading else None,
        "load_time": time.time() - model_loading_start_time if model_loading_start_time and model_loading else None
    })

@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 200
    
    # Check if model is loaded
    if not model_loaded:
        return jsonify({"error": "Model still loading, please wait..."}), 503
    
    try:
        # Debug environment
        logging.debug(f"PATH environment: {os.environ.get('PATH', 'Not set')}")
        logging.debug(f"Current working directory: {os.getcwd()}")
        
        logging.debug(f"Received request: {request.method} {request.path}")
        logging.debug(f"Request headers: {dict(request.headers)}")
        logging.debug(f"Request files: {request.files}")
        
        # Get audio file from request
        if 'audio' not in request.files:
            logging.error("No audio file in request")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
            
        # Check actual file size after saving
        file_size = os.path.getsize(tmp_path)
        logging.debug(f"Audio file: {audio_file.filename}, saved size: {file_size} bytes")
        logging.debug(f"Saved audio to: {tmp_path}")
        
        # Validate file size
        if file_size == 0:
            os.remove(tmp_path)
            logging.error("Received empty audio file")
            return jsonify({"error": "Received empty audio file"}), 400
        
        # Convert webm to wav using ffmpeg
        wav_path = tmp_path.replace('.webm', '.wav')
        try:
            # Try multiple methods to find ffmpeg
            ffmpeg_path = None
            
            # Method 1: Check common installation paths
            common_paths = [
                '/usr/local/bin/ffmpeg',
                '/opt/homebrew/bin/ffmpeg',  # Homebrew on Apple Silicon
                '/usr/bin/ffmpeg',
                '/opt/local/bin/ffmpeg',  # MacPorts
            ]
            
            for path in common_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    ffmpeg_path = path
                    logging.debug(f"Found ffmpeg at: {path}")
                    break
            
            # Method 2: Use shutil.which with expanded PATH
            if not ffmpeg_path:
                # Create an expanded PATH that includes common locations
                expanded_path = os.environ.get('PATH', '')
                extra_paths = ['/usr/local/bin', '/opt/homebrew/bin', '/usr/bin', '/opt/local/bin']
                for extra_path in extra_paths:
                    if extra_path not in expanded_path:
                        expanded_path = f"{extra_path}:{expanded_path}"
                
                # Create a custom environment with expanded PATH
                env = os.environ.copy()
                env['PATH'] = expanded_path
                
                # Try to find ffmpeg using which
                ffmpeg_path = shutil.which('ffmpeg', path=expanded_path)
                if ffmpeg_path:
                    logging.debug(f"Found ffmpeg via which: {ffmpeg_path}")
            
            if not ffmpeg_path:
                # Log what we tried
                logging.error(f"FFmpeg not found. Searched paths: {common_paths}")
                logging.error(f"PATH environment: {os.environ.get('PATH', 'Not set')}")
                logging.error(f"Expanded PATH: {expanded_path}")
                raise FileNotFoundError("FFmpeg not found in any expected location")
            
            logging.debug(f"Running ffmpeg command: {ffmpeg_path} -i {tmp_path} -ar 16000 -ac 1 -f wav {wav_path}")
            
            # Run with expanded PATH environment
            env = os.environ.copy()
            env['PATH'] = f"/usr/local/bin:/opt/homebrew/bin:{env.get('PATH', '')}"
            
            result = subprocess.run([
                ffmpeg_path, '-i', tmp_path, '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path
            ], check=True, capture_output=True, text=True, env=env)
            logging.debug(f"FFmpeg conversion successful")
            if result.stderr:
                logging.debug(f"FFmpeg stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg failed with return code {e.returncode}")
            logging.error(f"FFmpeg stderr: {e.stderr}")
            logging.error(f"FFmpeg stdout: {e.stdout}")
            os.remove(tmp_path)
            return jsonify({"error": f"Audio conversion failed: {e.stderr}"}), 500
        except FileNotFoundError as e:
            logging.error(f"FFmpeg not found: {e}")
            logging.error(f"Searched paths: {common_paths}")
            os.remove(tmp_path)
            return jsonify({"error": f"FFmpeg not found. Searched: {', '.join(common_paths)}. PATH: {os.environ.get('PATH', 'Not set')}"}), 500
        
        # Debug: Save files for inspection
        import shutil
        debug_dir = "/tmp/whisper_debug_audio"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(tmp_path, f"{debug_dir}/{timestamp}_input.webm")
        shutil.copy(wav_path, f"{debug_dir}/{timestamp}_output.wav")
        logging.info(f"Debug: Saved audio files to {debug_dir}/{timestamp}_*")
        
        # Transcribe
        try:
            result = model.transcribe(wav_path, language="en", fp16=False)
            text = result["text"].strip()
            logging.info(f"Transcription result: '{text}'")
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        
        return jsonify({"text": text})
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Unhandled error in transcribe: {error_details}")
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

def check_model_cache():
    """Check if model is already cached to provide better startup estimates"""
    try:
        import os
        import pathlib
        cache_dir = os.path.join(str(pathlib.Path.home()), '.cache', 'whisper')
        model_file = f"{WHISPER_MODEL}.pt"
        model_path = os.path.join(cache_dir, model_file)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logging.info(f"Model file found in cache: {model_path} ({size_mb:.1f} MB)")
            return True
        else:
            logging.info(f"Model file not found in cache, will download: {model_file}")
            return False
    except Exception as e:
        logging.warning(f"Could not check model cache: {e}")
        return False

if __name__ == '__main__':
    print(f"\n{'*'*60}")
    print(f"Whisper Transcribe Server Starting")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"{'*'*60}\n")
    
    # Check if model is cached
    is_cached = check_model_cache()
    if is_cached:
        print("✓ Model found in cache - startup will be faster")
    else:
        print("⚠ Model not cached - will download on first load")
        print(f"  Expected download size: {MODEL_SIZES.get(WHISPER_MODEL, 'Unknown')}")
    
    # Start loading model in a separate thread
    import threading
    model_thread = threading.Thread(target=load_model_async, daemon=True)
    model_thread.start()
    
    # Give a moment for thread to start
    time.sleep(0.1)
    
    # Disable request size limit
    app.config['MAX_CONTENT_LENGTH'] = None
    
    # Disable debug mode in production to prevent auto-reloading issues
    is_packaged = getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS') or 'Contents/Resources' in sys.executable
    
    print(f"\nServer starting on http://127.0.0.1:5001")
    print(f"Debug mode: {'OFF (production)' if is_packaged else 'ON (development)'}")
    print(f"Model loading in background...\n")
    
    # IMPORTANT: use_reloader=False prevents model from being loaded multiple times
    app.run(host='127.0.0.1', port=5001, debug=not is_packaged, threaded=True, use_reloader=False)