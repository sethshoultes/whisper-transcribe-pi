#!/usr/bin/env python3
"""
Preload Whisper Model Script

This script can be run to pre-download and cache the Whisper model
before starting the main application, reducing startup time.

Usage:
    python preload_model.py [model_name]
    
Example:
    python preload_model.py tiny
    python preload_model.py base
"""

import sys
import time
import whisper
from datetime import datetime
import os
import pathlib

def preload_model(model_name="tiny"):
    """Pre-download and cache a Whisper model"""
    
    print(f"\n{'='*60}")
    print(f"Whisper Model Preloader")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Check cache
    cache_dir = os.path.join(str(pathlib.Path.home()), '.cache', 'whisper')
    model_file = f"{model_name}.pt"
    model_path = os.path.join(cache_dir, model_file)
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model already cached: {model_path} ({size_mb:.1f} MB)")
        print("  Loading to verify...")
    else:
        print(f"⚠ Model not cached, will download...")
        print(f"  Cache location: {cache_dir}")
    
    # Load model (this will download if needed)
    start_time = time.time()
    try:
        print(f"\nLoading model '{model_name}'...")
        model = whisper.load_model(model_name)
        load_time = time.time() - start_time
        
        print(f"\n✓ Model loaded successfully in {load_time:.1f} seconds")
        
        # Test transcription
        print("\nPerforming test transcription...")
        test_start = time.time()
        import numpy as np
        silence = np.zeros(16000, dtype=np.float32)
        result = model.transcribe(silence, language="en", fp16=False)
        test_time = time.time() - test_start
        
        print(f"✓ Test transcription completed in {test_time:.1f} seconds")
        
        # Final check
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"\n✓ Model cached successfully: {size_mb:.1f} MB")
        
        print(f"\n{'='*60}")
        print(f"Model '{model_name}' is ready for use!")
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print(f"Time elapsed: {time.time() - start_time:.1f} seconds")
        return False

if __name__ == "__main__":
    # Get model name from command line or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    
    # Validate model name
    valid_models = ["tiny", "base", "small", "medium", "large"]
    if model_name not in valid_models:
        print(f"Error: Invalid model '{model_name}'")
        print(f"Valid models: {', '.join(valid_models)}")
        sys.exit(1)
    
    # Preload the model
    success = preload_model(model_name)
    sys.exit(0 if success else 1)