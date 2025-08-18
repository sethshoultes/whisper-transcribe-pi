"""
Whisper Model Configuration

This file defines available models and their characteristics.
The Electron app can query this information via the /model-info endpoint.
"""

# Available Whisper models with their characteristics
WHISPER_MODELS = {
    "tiny": {
        "size_mb": 39,
        "parameters": "39M",
        "english_only": False,
        "description": "Fastest, suitable for real-time on most devices",
        "relative_speed": 1.0,
        "memory_required_mb": 150,
        "download_time_estimate_seconds": 10  # On decent connection
    },
    "base": {
        "size_mb": 74,
        "parameters": "74M",
        "english_only": False,
        "description": "Good balance of speed and accuracy",
        "relative_speed": 0.5,  # 2x slower than tiny
        "memory_required_mb": 250,
        "download_time_estimate_seconds": 20
    },
    "small": {
        "size_mb": 244,
        "parameters": "244M",
        "english_only": False,
        "description": "Better accuracy, slower processing",
        "relative_speed": 0.2,  # 5x slower than tiny
        "memory_required_mb": 600,
        "download_time_estimate_seconds": 60
    },
    "medium": {
        "size_mb": 769,
        "parameters": "769M",
        "english_only": False,
        "description": "High accuracy, requires more resources",
        "relative_speed": 0.1,  # 10x slower than tiny
        "memory_required_mb": 1500,
        "download_time_estimate_seconds": 180
    },
    "large": {
        "size_mb": 1550,
        "parameters": "1550M",
        "english_only": False,
        "description": "Best accuracy, very slow on CPU",
        "relative_speed": 0.05,  # 20x slower than tiny
        "memory_required_mb": 3000,
        "download_time_estimate_seconds": 360
    }
}

# Model download URLs (for reference, handled by whisper library)
MODEL_URLS = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt"
}

# Default model recommendation based on system
def get_recommended_model():
    """Get recommended model based on available system resources"""
    try:
        import psutil
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Recommend based on available memory
        if available_memory_mb < 500:
            return "tiny"
        elif available_memory_mb < 1000:
            return "base"
        elif available_memory_mb < 2000:
            return "small"
        elif available_memory_mb < 4000:
            return "medium"
        else:
            return "large"
    except:
        # Default to tiny if we can't determine
        return "tiny"