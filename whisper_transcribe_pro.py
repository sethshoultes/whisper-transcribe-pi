#!/usr/bin/env python3
"""
Whisper Transcribe Pro - Enhanced version with modern UI and Hailo AI integration
Based on comprehensive GUI design documentation and improvement suggestions
"""

import os
import sys
import time
import threading
import queue
import logging
import json
import numpy as np
import sounddevice as sd
import whisper
import tempfile
import wave
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import customtkinter as ctk
from PIL import Image, ImageDraw
from scipy import signal
from scipy.signal import butter, lfilter

# Voice Memory Manager Integration
try:
    from tools.memory.voice_memory_manager import VoiceMemoryManager, create_voice_memory_manager
    VOICE_MEMORY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Voice Memory Manager not available: {e}")
    VOICE_MEMORY_AVAILABLE = False
    VoiceMemoryManager = None

# Configure appearance
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Set up logging
# Check if debug logging is enabled in settings
settings_path = os.path.expanduser("~/.whisper_transcribe_pro.json")
debug_enabled = False
if os.path.exists(settings_path):
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            debug_enabled = settings.get('debug_logging', False)
    except:
        pass

logging.basicConfig(
    level=logging.DEBUG if debug_enabled else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/whisper_pro.log'),
        logging.StreamHandler()
    ]
)

class AudioVisualizer:
    """Real-time audio waveform visualization"""
    
    def __init__(self, canvas_width=400, canvas_height=60):
        self.width = canvas_width
        self.height = canvas_height
        self.waveform_data = np.zeros(100)
        self.canvas = None  # Will be set by the main app
        
    def set_canvas(self, canvas):
        """Set the tkinter canvas for drawing"""
        self.canvas = canvas
        
    def update_level(self, rms_value):
        """Update visualization with audio level"""
        if self.canvas is not None:
            try:
                # Add to rolling buffer
                if not hasattr(self, 'rms_history'):
                    self.rms_history = []
                    
                self.rms_history.append(rms_value)
                if len(self.rms_history) > 50:  # Keep last 50 samples
                    self.rms_history.pop(0)
                
                # Update and draw
                self.draw_level_bars()
            except Exception as e:
                logging.debug(f"Visualizer update error: {e}")
    
    def draw_level_bars(self):
        """Draw audio level bars"""
        if self.canvas is None or not hasattr(self, 'rms_history'):
            return
            
        try:
            self.canvas.delete("all")
            
            width = self.width
            height = self.height
            mid_y = height // 2
            
            # Draw bars for each RMS value
            bar_width = width / 50
            for i, rms in enumerate(self.rms_history):
                # Scale RMS to pixel height (typical range 0.0 to 0.1)
                bar_height = min(int(rms * 500), height // 2 - 2)
                x = i * bar_width
                
                # Color based on level
                if rms > 0.05:
                    color = "#22C55E"  # Green for loud
                elif rms > 0.01:
                    color = "#4299E1"  # Blue for normal
                else:
                    color = "#64748B"  # Gray for quiet
                
                # Draw symmetric bars
                self.canvas.create_rectangle(
                    x, mid_y - bar_height,
                    x + bar_width - 1, mid_y + bar_height,
                    fill=color, outline=""
                )
            
            # Center line
            self.canvas.create_line(0, mid_y, width, mid_y, fill="#2D3748", width=1)
        except Exception as e:
            logging.debug(f"Canvas draw error: {e}")
    
    def draw_waveform(self):
        """Draw the waveform on the canvas"""
        if self.canvas is None:
            return
            
        try:
            # Clear canvas
            self.canvas.delete("all")
            
            # Calculate dimensions
            width = self.width
            height = self.height
            mid_y = height // 2
            
            # Normalize waveform data with fixed scale for consistency
            # Use a fixed scale so quiet sounds still show
            normalized = np.clip(self.waveform_data * 50, 0, 1)  # Scale up quiet sounds
            
            # Draw waveform bars
            bar_width = max(2, width / len(normalized))
            for i, value in enumerate(normalized):
                # Calculate bar height (symmetric around center)
                bar_height = int(value * (height // 2 - 5))
                x = i * bar_width
                
                # Draw bar from center
                self.canvas.create_rectangle(
                    x, mid_y - bar_height,
                    x + bar_width - 1, mid_y + bar_height,
                    fill="#4299E1", outline=""
                )
            
            # Draw center line
            self.canvas.create_line(0, mid_y, width, mid_y, fill="#2D3748", width=1)
        except Exception as e:
            logging.debug(f"Canvas draw error: {e}")

class HailoIntegration:
    """Optional Hailo AI integration for enhanced audio processing"""
    
    def __init__(self):
        self.hailo_available = self.check_hailo()
        self.audio_enhancement_enabled = False  # Will be controlled by settings
        
    def check_hailo(self) -> bool:
        """Check if Hailo is available on the system"""
        try:
            # Check for hailortcli
            result = subprocess.run(
                ['which', 'hailortcli'],
                capture_output=True,
                text=True
            )
            # Also verify the Hailo detection script exists
            hailo_script = os.path.expanduser("~/hailo-ai/scripts/simple_photo_detect.sh")
            if result.returncode == 0 and os.path.exists(hailo_script):
                logging.info("Hailo AI detected - enhanced features available")
                return True
        except:
            pass
        return False
    
    def enhance_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Use Hailo AI for fast audio enhancement"""
        if not self.hailo_available or not self.audio_enhancement_enabled:
            return audio_data
            
        try:
            # Simplified, fast audio enhancement
            enhanced = audio_data.copy()
            
            # 1. Fast noise gate (removes very quiet parts)
            if len(enhanced) > 0:
                # Simple but effective noise gate
                threshold = np.percentile(np.abs(enhanced), 15)  # Bottom 15% is noise
                gate_mask = np.abs(enhanced) > threshold
                
                # Smooth the mask to avoid clicks
                from scipy.ndimage import uniform_filter1d
                gate_mask = uniform_filter1d(gate_mask.astype(float), size=100) > 0.5
                enhanced = enhanced * gate_mask
            
            # 2. Simple automatic gain control (normalize volume)
            if len(enhanced) > 0:
                # Fast AGC - normalize to optimal range
                max_val = np.max(np.abs(enhanced))
                if max_val > 0.001:  # Avoid dividing by zero
                    # Normalize to 80% of maximum to avoid clipping
                    enhanced = enhanced * (0.8 / max_val)
            
            # 3. Quick speech frequency emphasis (optional, very fast)
            # Only apply if really needed - this is the most expensive operation
            if self.audio_enhancement_enabled and sample_rate > 8000:
                from scipy.signal import butter, lfilter
                # Simple high-shelf filter to boost speech clarity
                nyquist = sample_rate / 2
                cutoff = 2000 / nyquist
                if cutoff < 1.0:
                    b, a = butter(1, cutoff, btype='high')
                    # Mix original with filtered for subtle enhancement
                    filtered = lfilter(b, a, enhanced)
                    enhanced = enhanced * 0.7 + filtered * 0.3
            
            logging.debug("Hailo audio enhancement applied (fast mode)")
            return enhanced.astype(np.float32)
            
        except Exception as e:
            logging.warning(f"Hailo audio enhancement failed: {e}")
            return audio_data

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def send_message(self, text: str) -> str:
        """Send text message and get response"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is ready and available"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return provider name"""
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Return current status description"""
        pass

class ClaudeAPIProvider(AIProvider):
    """Claude API provider using Anthropic's API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = "claude-3-haiku-20240307"
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the Anthropic client"""
        if not self.api_key:
            logging.warning("Claude API key not found. Set ANTHROPIC_API_KEY environment variable.")
            return
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logging.info("Claude API client initialized successfully")
        except ImportError:
            logging.error("anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            logging.error(f"Failed to initialize Claude API client: {e}")
    
    def send_message(self, text: str) -> str:
        """Send message to Claude API"""
        if not self.client:
            return "Error: Claude API not available. Check API key and internet connection."
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            return response.content[0].text
        
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                return "Error: Rate limit exceeded. Please wait before trying again."
            elif "invalid api key" in error_msg or "authentication" in error_msg:
                return "Error: Invalid API key. Please check your Anthropic API key."
            elif "network" in error_msg or "connection" in error_msg:
                return "Error: Network connection failed. Check your internet connection."
            else:
                return f"Error: Claude API request failed - {e}"
    
    def is_available(self) -> bool:
        """Check if Claude API is available"""
        return self.client is not None and self.api_key is not None
    
    def get_name(self) -> str:
        """Return provider name"""
        return "Claude API (Haiku)"
    
    def get_status(self) -> str:
        """Return current status"""
        if not self.api_key:
            return "Missing API key"
        elif not self.client:
            return "Client setup failed"
        else:
            return "Ready"

class OpenAIProvider(AIProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the OpenAI client"""
        if not self.api_key:
            logging.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logging.info(f"OpenAI API client initialized successfully with model {self.model}")
        except ImportError:
            logging.error("openai library not installed. Run: pip install openai")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI API client: {e}")
    
    def send_message(self, text: str) -> str:
        """Send message to OpenAI API"""
        if not self.client:
            return "Error: OpenAI API not available. Check API key and internet connection."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": text}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                return "Error: Rate limit exceeded. Please wait before trying again."
            elif "invalid api key" in error_msg or "authentication" in error_msg:
                return "Error: Invalid API key. Please check your OpenAI API key."
            elif "network" in error_msg or "connection" in error_msg:
                return "Error: Network connection failed. Check your internet connection."
            elif "model" in error_msg and "not found" in error_msg:
                return f"Error: Model {self.model} not available. Try gpt-3.5-turbo or gpt-4."
            else:
                return f"Error: OpenAI API request failed - {e}"
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return self.client is not None and self.api_key is not None
    
    def get_name(self) -> str:
        """Return provider name"""
        return f"OpenAI ({self.model})"
    
    def get_status(self) -> str:
        """Return current status"""
        if not self.api_key:
            return "Missing API key"
        elif not self.client:
            return "Client setup failed"
        else:
            return f"Ready ({self.model})"
    
    def set_model(self, model: str):
        """Change the model used by this provider"""
        self.model = model
        logging.info(f"OpenAI model changed to {model}")

class LocalAI(AIProvider):
    """Manage local LLM server integration"""
    
    def __init__(self, settings):
        self.settings = settings
        self.server_process = None
        self.llm_dir = Path.home() / "simple-llm-server"
        self.models_dir = self.llm_dir / "models"
        self.current_model = None
        
        # TinyLlama model info
        self.tinyllama_info = {
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size_mb": 638  # Approximate size in MB
        }
        
    def is_enabled(self):
        """Check if AI is enabled in settings"""
        return self.settings.settings.get("ai_enabled", False)
    
    def get_available_models(self):
        """Get list of available .gguf models"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_file in self.models_dir.glob("*.gguf"):
            # Check if it's a real file (not a broken symlink)
            if model_file.exists() and model_file.stat().st_size > 1024*1024:  # > 1MB
                models.append(model_file.name)
        return sorted(models)
    
    def is_tinyllama_available(self):
        """Check if TinyLlama model is available locally"""
        model_path = self.models_dir / self.tinyllama_info["filename"]
        # Check if file exists and is not a broken symlink
        try:
            return model_path.exists() and model_path.stat().st_size > 100*1024*1024  # > 100MB
        except:
            return False
    
    def download_tinyllama(self, progress_callback=None):
        """Download TinyLlama model with progress tracking"""
        import requests
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = self.models_dir / self.tinyllama_info["filename"]
        url = self.tinyllama_info["url"]
        
        try:
            # Start download with streaming
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Write to file with progress updates
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)
            
            return True
        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            logging.error(f"Failed to download TinyLlama: {e}")
            return False
    
    def start_server(self, model_name=None):
        """Start the local LLM server"""
        # If trying to start TinyLlama and it's not available, return False
        if model_name and model_name == self.tinyllama_info["filename"]:
            if not self.is_tinyllama_available():
                logging.warning("TinyLlama not available locally, cannot start server")
                return False
        
        # Check if any server is running (ours or external)
        if self.is_server_running():
            # Server already running, need to restart with new model
            self.stop_server()
            import time
            time.sleep(2)  # Give more time for shutdown
            
        try:
            env_dir = self.llm_dir / "env"
            if not env_dir.exists():
                return False
                
            # Use llm_api_server.py with model parameter
            cmd = [
                str(env_dir / "bin" / "python"),
                str(self.llm_dir / "llm_api_server.py")
            ]
            
            # Add model parameter if specified
            if model_name:
                cmd.extend(["--model", model_name])
            
            # Start server in background
            self.server_process = subprocess.Popen(
                cmd,
                cwd=str(self.llm_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it a moment to start
            import time
            time.sleep(3)  # Give more time for model loading
            
            if self.server_process.poll() is None:
                self.current_model = model_name if model_name else "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
                return True
            else:
                return False
            
        except Exception as e:
            logging.error(f"Failed to start AI server: {e}")
            return False
    
    def stop_server(self):
        """Stop the local LLM server"""
        # First try to stop our tracked process
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
        
        # Also kill any other llm_api_server processes on port 7861
        try:
            # Find and kill any process using port 7861
            result = subprocess.run(
                ["lsof", "-ti", ":7861"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", pid], check=False)
                    except:
                        pass
        except:
            pass
    
    def is_server_running(self):
        """Check if server is running"""
        if self.server_process and self.server_process.poll() is None:
            return True
        
        # Check if simple_api.py is running on port 7861
        try:
            import requests
            response = requests.get("http://localhost:7861/health", timeout=2)
            if response.status_code == 200:
                # Get the actual model from the server
                data = response.json()
                if 'model' in data:
                    # Extract just the model filename from the path
                    model_path = data['model']
                    self.current_model = model_path.split('/')[-1] if '/' in model_path else model_path
                return True
            return False
        except:
            return False
    
    def get_current_model(self):
        """Get the currently loaded model name"""
        if self.is_server_running() and self.current_model:
            # Return shortened name for display
            model_map = {
                "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": "TinyLlama",
                "phi-2.Q4_K_M.gguf": "Phi-2",
                "mistral-7b-instruct-v0.2.Q4_K_M.gguf": "Mistral-7B"
            }
            return model_map.get(self.current_model, self.current_model.replace(".gguf", ""))
        return None
    
    def send_to_ai(self, text):
        """Send text to local AI and get response"""
        if not self.is_server_running():
            return {"success": False, "error": "AI server not running"}
            
        try:
            import requests
            # Call simple_api.py endpoint
            response = requests.post(
                "http://localhost:7861/api/generate",
                json={
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "No response")
                return {"success": True, "response": ai_response}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # AIProvider interface implementation
    def send_message(self, text: str) -> str:
        """Send message to local AI server (AIProvider interface)"""
        result = self.send_to_ai(text)
        if result["success"]:
            return result["response"]
        else:
            return f"Error: {result['error']}"
    
    def is_available(self) -> bool:
        """Check if local AI is available (AIProvider interface)"""
        return self.is_server_running()
    
    def get_name(self) -> str:
        """Return provider name (AIProvider interface)"""
        current_model = self.get_current_model()
        if current_model:
            return f"Local AI ({current_model})"
        else:
            return "Local AI"
    
    def get_status(self) -> str:
        """Return current status (AIProvider interface)"""
        if not self.is_enabled():
            return "Disabled in settings"
        elif not self.is_server_running():
            return "Server not running"
        else:
            model = self.get_current_model()
            return f"Running ({model})" if model else "Running"

class Settings:
    """Application settings management"""
    
    def __init__(self):
        self.config_path = Path.home() / ".whisper_transcribe_pro.json"
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults"""
        defaults = {
            "theme": "dark",
            "model": "tiny",
            "language": "en",
            "always_on_top": True,
            "font_size": 12,
            "noise_reduction": False,
            "vad_enabled": False,
            "hailo_integration": False,
            "compact_mode": False,
            "window_size": "standard",  # compact, standard, large
            "audio_device": None,
            "sample_rate": 44100,
            "hotkeys": {
                "record": "r",
                "stop": "s",
                "clear": "c",
                "copy": "ctrl+c"
            },
            "ai_enabled": False,
            "ai_auto_send": False,
            "ai_provider": "local",
            "claude_api_key": "",
            "openai_api_key": "",
            "openai_model": "gpt-3.5-turbo",
            # Voice Memory Settings
            "voice_memory_enabled": True,
            "voice_memory_context_limit": 10,
            "voice_memory_audio_metadata": True,
            "voice_memory_pattern_learning": True,
            "voice_memory_wake_word_tracking": True,
            "voice_memory_db_path": "data/voice_conversations.db",
            "voice_memory_auto_save": True,
            "voice_memory_compression": True
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    saved = json.load(f)
                    defaults.update(saved)
            except:
                pass
        
        return defaults
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")

class WhisperTranscribePro(ctk.CTk):
    """Main application class with enhanced UI and features"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.settings = Settings()
        self.hailo = HailoIntegration()
        self.visualizer = AudioVisualizer()
        self.local_ai = LocalAI(self.settings)
        
        # Initialize Voice Memory Manager
        self.voice_memory = None
        self._init_voice_memory()
        
        # Audio/transcription state
        self.model = None
        self.recording = False
        self.processing = False
        self.audio_data = []
        self.device_sample_rate = 44100
        self.whisper_sample_rate = 16000
        self.transcription_history = []
        self.ui_queue = queue.Queue()
        
        # Find audio device
        self.setup_audio_device()
        
        # Setup UI
        self.setup_window()
        self.create_widgets()
        self.apply_theme()
        
        # Load Whisper model in background
        self.load_model_thread = threading.Thread(target=self.load_model, daemon=True)
        self.load_model_thread.start()
        
        # Start UI update loop
        self.process_ui_queue()
        
        # Start periodic AI status check
        self.check_ai_status()
    
    def _init_voice_memory(self):
        """Initialize Voice Memory Manager with error handling"""
        if not VOICE_MEMORY_AVAILABLE or not self.settings.settings.get("voice_memory_enabled", True):
            logging.info("Voice Memory Manager disabled or not available")
            return
        
        try:
            # Create memory configuration from settings
            memory_config = {
                'enable_context_memory': True,
                'enable_audio_metadata': self.settings.settings.get("voice_memory_audio_metadata", True),
                'conversation_memory_limit': self.settings.settings.get("voice_memory_context_limit", 100),
                'conversation_db_path': self.settings.settings.get("voice_memory_db_path", "data/voice_conversations.db"),
                'enable_pattern_learning': self.settings.settings.get("voice_memory_pattern_learning", True),
                'enable_memory_compression': self.settings.settings.get("voice_memory_compression", True),
                'auto_save_interval': 300 if self.settings.settings.get("voice_memory_auto_save", True) else 0,
                'wake_word_learning_enabled': self.settings.settings.get("voice_memory_wake_word_tracking", True)
            }
            
            # Ensure data directory exists
            db_path = Path(memory_config['conversation_db_path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize Voice Memory Manager
            self.voice_memory = create_voice_memory_manager(memory_config)
            logging.info("Voice Memory Manager initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Voice Memory Manager: {e}")
            self.voice_memory = None
    
    def cleanup_voice_memory(self):
        """Clean up voice memory resources on application close"""
        if hasattr(self, 'voice_memory') and self.voice_memory:
            try:
                logging.info("Cleaning up Voice Memory Manager...")
                self.voice_memory.close()
                self.voice_memory = None
                logging.info("Voice Memory Manager cleanup completed")
            except Exception as e:
                logging.error(f"Error during voice memory cleanup: {e}")
    
    def on_closing(self):
        """Handle application closing with proper cleanup"""
        try:
            # Save settings
            if hasattr(self, 'settings'):
                self.settings.save_settings()
            
            # Cleanup voice memory
            self.cleanup_voice_memory()
            
            # Close the application
            self.destroy()
            
        except Exception as e:
            logging.error(f"Error during application cleanup: {e}")
            self.destroy()  # Force close if cleanup fails
    
    def setup_audio_device(self):
        """Configure audio input device"""
        devices = sd.query_devices()
        self.device_index = None
        
        # Priority: USB mic > default input > any input
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if 'usb' in device['name'].lower():
                    self.device_index = i
                    self.device_sample_rate = int(device['default_samplerate'])
                    logging.info(f"Using USB mic: {device['name']} @ {self.device_sample_rate}Hz")
                    break
        
        if self.device_index is None:
            try:
                self.device_index = sd.default.device[0]
                default_device = devices[self.device_index]
                self.device_sample_rate = int(default_device['default_samplerate'])
                logging.info(f"Using default mic: {default_device['name']}")
            except:
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        self.device_index = i
                        self.device_sample_rate = int(device['default_samplerate'])
                        break
    
    def setup_window(self):
        """Configure main window properties"""
        self.title("Whisper Transcribe Pro")
        
        # Window size based on settings
        sizes = {
            "compact": "400x300",
            "standard": "600x500",
            "large": "800x600"
        }
        self.geometry(sizes.get(self.settings.settings["window_size"], "600x500"))
        
        # Always on top
        if self.settings.settings["always_on_top"]:
            self.attributes("-topmost", True)
        
        # Position window
        self.geometry("+50+50")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Register cleanup handler for window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create all UI elements with modern design"""
        
        # Top Frame - Status Bar
        self.top_frame = ctk.CTkFrame(self, height=40, corner_radius=0)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.top_frame.grid_columnconfigure(1, weight=1)
        
        # App Title
        self.title_label = ctk.CTkLabel(
            self.top_frame,
            text=" Whisper Transcribe Pro",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Status Indicator
        self.status_label = ctk.CTkLabel(
            self.top_frame,
            text="* Ready",
            font=ctk.CTkFont(size=12),
            text_color="green"
        )
        self.status_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        
        # Settings Button
        self.settings_btn = ctk.CTkButton(
            self.top_frame,
            text="Settings",
            width=30,
            command=self.open_settings
        )
        self.settings_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Main Content Frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)
        
        # Audio Visualizer Canvas (hidden initially)
        self.visualizer_frame = ctk.CTkFrame(self.main_frame, height=60)
        
        # Create actual tkinter canvas for waveform drawing
        import tkinter as tk
        self.waveform_canvas = tk.Canvas(
            self.visualizer_frame,
            width=580,  # Match frame width minus padding
            height=60,
            bg='#1a1a1a' if self.settings.settings.get("theme") == "dark" else '#f0f0f0',
            highlightthickness=0
        )
        self.waveform_canvas.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Connect canvas to visualizer
        self.visualizer.set_canvas(self.waveform_canvas)
        self.visualizer.width = 580
        
        # Will be shown during recording
        
        # Record Button - Large and prominent
        self.record_button = ctk.CTkButton(
            self.main_frame,
            text="Click to Record",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            corner_radius=30,
            fg_color="#4299E1",
            hover_color="#3182CE",
            command=self.toggle_recording
        )
        self.record_button.grid(row=1, column=0, padx=20, pady=15, sticky="ew")
        
        # Dual Panel Layout: Transcriptions + AI Responses
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=2)  # Transcription panel (wider)
        self.content_frame.grid_columnconfigure(1, weight=1)  # AI panel (narrower)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Transcription Panel
        self.text_frame = ctk.CTkFrame(self.content_frame)
        self.text_frame.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        self.text_frame.grid_columnconfigure(0, weight=1)
        self.text_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            self.text_frame,
            text="Transcriptions",
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, pady=5)
        
        self.text_display = ctk.CTkTextbox(
            self.text_frame,
            font=ctk.CTkFont(size=self.settings.settings["font_size"]),
            wrap="word"
        )
        self.text_display.grid(row=1, column=0, sticky="nsew")
        
        # AI Response Panel (initially hidden if AI not enabled)
        self.ai_frame = ctk.CTkFrame(self.content_frame)
        self.ai_frame.grid_columnconfigure(0, weight=1)
        self.ai_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            self.ai_frame,
            text="AI Responses",
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, pady=5)
        
        self.ai_display = ctk.CTkTextbox(
            self.ai_frame,
            font=ctk.CTkFont(size=self.settings.settings["font_size"]),
            wrap="word"
        )
        self.ai_display.grid(row=1, column=0, sticky="nsew")
        
        # Action Buttons Frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure((0,1,2,3,4,5,6), weight=1)
        
        # Action Buttons (without emoji for Pi compatibility)
        buttons = [
            ("Copy Last", self.copy_last),
            ("Copy All", self.copy_all),
            ("Export", self.export_transcription),
            ("Send to AI", self.send_to_ai),
            ("Memory", self.open_memory_menu),
            ("Search", self.search_transcription),
            ("Clear", self.clear_transcriptions)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ctk.CTkButton(
                self.button_frame,
                text=text,
                command=command,
                height=35,
                font=ctk.CTkFont(size=11)
            )
            btn.grid(row=0, column=i, padx=2, pady=0, sticky="ew")
        
        # Bottom Status Bar
        self.bottom_frame = ctk.CTkFrame(self, height=25, corner_radius=0)
        self.bottom_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        
        # Info labels
        self.model_label = ctk.CTkLabel(
            self.bottom_frame,
            text=f"Whisper: {self.settings.settings['model']}",
            font=ctk.CTkFont(size=10)
        )
        self.model_label.grid(row=0, column=0, padx=5, pady=2)
        
        # AI Model label
        self.ai_model_label = ctk.CTkLabel(
            self.bottom_frame,
            text="AI: Not Running",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.ai_model_label.grid(row=0, column=1, padx=5, pady=2)
        
        # Hailo status label (shows actual enabled state, not just availability)
        hailo_status = "Hailo: ON" if (self.hailo.hailo_available and self.settings.settings.get("hailo_integration", False)) else "Hailo: OFF"
        hailo_color = "green" if (self.hailo.hailo_available and self.settings.settings.get("hailo_integration", False)) else "gray"
        
        self.hailo_label = ctk.CTkLabel(
            self.bottom_frame,
            text=hailo_status,
            font=ctk.CTkFont(size=10),
            text_color=hailo_color
        )
        self.hailo_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Voice Memory status label
        self.update_voice_memory_status_label()
        
        # Show/hide AI panel based on setting (after all UI elements are created)
        self.update_ai_panel_visibility()
        
        # Keyboard bindings
        self.bind('<r>', lambda e: self.toggle_recording())
        self.bind('<R>', lambda e: self.toggle_recording())
        self.bind('<space>', lambda e: self.toggle_recording())
        self.bind('<Escape>', lambda e: self.quit())
    
    def apply_theme(self):
        """Apply theme settings"""
        theme = self.settings.settings["theme"]
        if theme in ["dark", "light"]:
            ctk.set_appearance_mode(theme)
    
    def load_model(self):
        """Load Whisper model in background"""
        self.update_status("Loading model...", "orange")
        try:
            model_name = self.settings.settings["model"]
            self.model = whisper.load_model(model_name)
            self.update_status("* Ready", "green")
            
            # Update model label in UI to show current model
            if hasattr(self, 'model_label'):
                self.ui_queue.put({
                    'type': 'model_update',
                    'model': model_name
                })
            
            logging.info(f"Loaded Whisper model: {model_name}")
        except Exception as e:
            self.update_status("Model load failed", "red")
            logging.error(f"Failed to load model: {e}")
    
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.model:
            self.show_notification("Model still loading...")
            return
        
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording"""
        self.recording = True
        self.audio_data = []
        
        # Update UI
        self.record_button.configure(
            text="Stop Recording",
            fg_color="#EF4444",
            hover_color="#DC2626"
        )
        self.update_status("* Recording...", "red")
        
        # Show visualizer
        self.visualizer_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.record_thread.start()
    
    def apply_noise_reduction(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction using high-pass filter"""
        if not self.settings.settings.get("noise_reduction", False):
            return audio_data
            
        try:
            # Simple noise reduction - don't filter out actual speech
            # Just remove very low frequencies
            nyquist = sample_rate / 2
            cutoff = 50 / nyquist  # Lower cutoff to preserve more speech
            if cutoff < 1.0:
                b, a = butter(2, cutoff, btype='high')  # Gentler filter
                filtered = lfilter(b, a, audio_data)
                return filtered
            return audio_data
        except Exception as e:
            logging.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains voice activity"""
        if not self.settings.settings.get("vad_enabled", False):
            return True
            
        try:
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            
            # Initialize history if needed
            if not hasattr(self, 'energy_history'):
                self.energy_history = []
            
            self.energy_history.append(energy)
            if len(self.energy_history) > 100:
                self.energy_history.pop(0)
            
            # Dynamic threshold
            if len(self.energy_history) > 10:
                noise_floor = np.percentile(self.energy_history, 20)
                voice_threshold = max(noise_floor * 3, 0.01)
                return energy > voice_threshold
            
            return energy > 0.01
            
        except Exception as e:
            logging.warning(f"VAD failed: {e}")
            return True
    
    def record_audio(self):
        """Record audio in background thread"""
        try:
            stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.device_sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=512
            )
            
            stream.start()
            
            while self.recording:
                try:
                    audio_chunk = stream.read(512)[0]
                    self.audio_data.append(audio_chunk)
                    
                    # Update visualizer with actual audio level
                    if len(self.audio_data) % 2 == 0:
                        # Calculate RMS for visualization
                        rms = np.sqrt(np.mean(audio_chunk ** 2))
                        # Pass RMS value for visualization
                        self.visualizer.update_level(rms)
                        
                except Exception as e:
                    if "PaErrorCode -9999" in str(e):
                        time.sleep(0.01)
                        continue
                    else:
                        break
            
            stream.stop()
            stream.close()
            
            # Process recording
            if len(self.audio_data) > 5:
                audio_array = np.concatenate(self.audio_data, axis=0).flatten()
                
                # Apply Hailo audio enhancement if enabled
                if self.hailo.hailo_available and self.settings.settings.get("hailo_integration", False):
                    self.hailo.audio_enhancement_enabled = True
                    audio_array = self.hailo.enhance_audio(audio_array, self.device_sample_rate)
                
                self.transcribe_audio(audio_array)
            else:
                self.update_status("Recording too short", "orange")
                
        except Exception as e:
            logging.error(f"Recording error: {e}")
            self.update_status("Recording error", "red")
    
    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False
        
        # Update UI
        self.record_button.configure(
            text="Click to Record",
            fg_color="#4299E1",
            hover_color="#3182CE"
        )
        self.update_status("Processing...", "orange")
        
        # Hide visualizer
        self.visualizer_frame.grid_remove()
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        # Check if model is loaded
        if self.model is None:
            self.update_status("Model not loaded", "red")
            return
            
        self.processing = True
        
        try:
            # Resample if needed
            if self.device_sample_rate != self.whisper_sample_rate:
                from scipy import signal
                num_samples = int(len(audio_data) * self.whisper_sample_rate / self.device_sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.whisper_sample_rate)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
                
                # Transcribe with current settings
                language = self.settings.settings.get("language", "en")
                result = self.model.transcribe(
                    tmp.name,
                    language=language,
                    fp16=False
                )
                
                text = result["text"].strip()
                
                if text:
                    # Add to history
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    entry = f"[{timestamp}] {text}\n\n"
                    self.transcription_history.append(text)
                    
                    # Store in voice memory
                    self._store_voice_transcription(text, result, tmp.name, audio_data)
                    
                    # Update display
                    self.ui_queue.put({
                        'type': 'transcription',
                        'text': entry
                    })
                    
                    # Auto-send to AI if enabled
                    if (self.local_ai.is_enabled() and 
                        self.settings.settings.get("ai_auto_send", False)):
                        self.auto_send_to_ai(text)
                    
                    # Copy to clipboard
                    self.copy_to_clipboard(text)
                    
                    # Check for Hailo integration (disabled automatic camera usage)
                    # Camera-based speaker detection is now manual-only to avoid privacy concerns
                    # Uncomment below to enable automatic speaker detection with camera
                    # if self.hailo.hailo_available and self.settings.settings["hailo_integration"]:
                    #     speaker = self.hailo.detect_speaker()
                    #     if speaker:
                    #         self.ui_queue.put({
                    #             'type': 'info',
                    #             'text': f"[Hailo: {speaker}]\n"
                    #         })
                    
                    self.update_status("* Ready", "green")
                else:
                    self.update_status("No speech detected", "orange")
                
                os.unlink(tmp.name)
                
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            self.update_status("Transcription error", "red")
        
        self.processing = False
    
    def _store_voice_transcription(self, text: str, result: dict, audio_file_path: str, audio_data: np.ndarray):
        """Store voice transcription in memory with metadata"""
        if not self.voice_memory:
            logging.debug("Voice memory not available, skipping transcription storage")
            return
        
        try:
            # Calculate audio metadata with fallbacks
            try:
                audio_duration = len(audio_data) / self.whisper_sample_rate
            except (ZeroDivisionError, TypeError):
                audio_duration = 0.0
                logging.warning("Could not calculate audio duration, using 0.0")
            
            try:
                confidence_score = self._calculate_confidence(result)
            except Exception as conf_error:
                confidence_score = 0.8
                logging.warning(f"Could not calculate confidence score: {conf_error}")
            
            audio_metadata = {
                'audio_file_path': audio_file_path,
                'confidence_score': confidence_score,
                'audio_duration': audio_duration,
                'processing_time': time.time(),
                'language': result.get('language', 'en') if result else 'en',
                'segments': result.get('segments', []) if result else []
            }
            
            # Store as pending transcription with graceful fallback
            self._pending_transcription = {
                'text': text,
                'audio_metadata': audio_metadata,
                'timestamp': time.time()
            }
            
            # IMPORTANT: Also immediately save to memory even without AI response
            # This ensures transcriptions are saved even when AI is disabled
            try:
                self.voice_memory.add_voice_interaction(
                    user_input=text,
                    assistant_response="",  # No AI response yet
                    audio_metadata=audio_metadata,
                    processing_times={'transcription': time.time() - audio_metadata.get('processing_time', time.time())},
                    is_transcription_only=True  # Flag to indicate this is transcription without AI
                )
                logging.debug("Voice transcription immediately saved to memory")
            except Exception as save_error:
                logging.error(f"Failed to immediately save transcription: {save_error}")
            
            logging.debug("Voice transcription stored in pending memory")
            
        except Exception as e:
            logging.error(f"Failed to store voice transcription (graceful fallback applied): {e}")
            # Graceful fallback: store minimal information
            self._pending_transcription = {
                'text': text,
                'audio_metadata': {'confidence_score': 0.8, 'audio_duration': 0.0},
                'timestamp': time.time()
            }
    
    def _store_ai_interaction(self, user_input: str, assistant_response: str):
        """Store complete AI interaction in voice memory with graceful fallbacks"""
        if not self.voice_memory:
            logging.debug("Voice memory not available, skipping AI interaction storage")
            return
        
        try:
            # Get pending transcription metadata if available
            audio_metadata = None
            try:
                if hasattr(self, '_pending_transcription') and self._pending_transcription:
                    if self._pending_transcription.get('text') == user_input:
                        audio_metadata = self._pending_transcription.get('audio_metadata', {})
                        # Clear pending transcription
                        self._pending_transcription = None
            except Exception as meta_error:
                logging.warning(f"Could not retrieve audio metadata: {meta_error}")
                audio_metadata = None
            
            # Store complete interaction with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.voice_memory.add_voice_interaction(
                        user_input=user_input,
                        assistant_response=assistant_response,
                        audio_metadata=audio_metadata,
                        agent_used="whisper_transcribe_pro",
                        command_type="transcription_request",
                        success=True
                    )
                    
                    conversation_id = result.get('conversation_id', 'unknown')
                    logging.debug(f"Stored voice interaction: {conversation_id}")
                    
                    # Check if storage was successful
                    if result.get('success', False):
                        return  # Success, exit retry loop
                    else:
                        logging.warning(f"Voice interaction storage reported failure: {result.get('errors', [])}")
                        
                except Exception as storage_error:
                    if attempt < max_retries - 1:
                        logging.warning(f"AI interaction storage attempt {attempt + 1} failed: {storage_error}")
                    else:
                        raise storage_error
            
        except Exception as e:
            logging.error(f"Failed to store AI interaction after all retries (graceful fallback applied): {e}")
            # Graceful fallback: log the interaction for potential manual recovery
            try:
                fallback_log = {
                    'timestamp': time.time(),
                    'user_input': user_input[:200],  # Truncate to prevent log spam
                    'assistant_response': assistant_response[:200],
                    'error': str(e)
                }
                logging.info(f"FALLBACK_INTERACTION: {fallback_log}")
            except Exception as fallback_error:
                logging.error(f"Even fallback logging failed: {fallback_error}")
    
    def update_voice_memory_status_label(self):
        """Update the voice memory status label in the bottom frame"""
        if hasattr(self, 'voice_memory') and self.voice_memory:
            memory_status = "Memory: ON"
            memory_color = "green"
        elif VOICE_MEMORY_AVAILABLE:
            memory_status = "Memory: OFF"
            memory_color = "orange"
        else:
            memory_status = "Memory: N/A"
            memory_color = "gray"
        
        if not hasattr(self, 'memory_label'):
            self.memory_label = ctk.CTkLabel(
                self.bottom_frame,
                text=memory_status,
                font=ctk.CTkFont(size=10),
                text_color=memory_color
            )
            self.memory_label.grid(row=0, column=3, padx=5, pady=2)
        else:
            self.memory_label.configure(text=memory_status, text_color=memory_color)
    
    def open_memory_menu(self):
        """Open voice memory management menu with conversation history"""
        logging.info("Memory menu button clicked - starting to open dialog")
        
        if not VOICE_MEMORY_AVAILABLE:
            logging.error("Voice Memory system not available")
            self.show_notification("Voice Memory system not available")
            return
        
        logging.info("Voice Memory is available, creating dialog window")
        
        # Create memory menu window
        memory_window = ctk.CTkToplevel(self)
        memory_window.title("Voice Memory Management")
        memory_window.geometry("750x600")  # Slightly narrower
        memory_window.transient(self)
        
        # Center the window
        memory_window.update_idletasks()
        x = (memory_window.winfo_screenwidth() // 2) - 375
        y = (memory_window.winfo_screenheight() // 2) - 300
        memory_window.geometry(f"750x600+{x}+{y}")
        
        logging.info("Memory dialog window created successfully")
        
        # Header frame with title and close button
        header_frame = ctk.CTkFrame(memory_window)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            header_frame,
            text="Voice Memory Management",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            header_frame,
            text="✕ Close",
            command=memory_window.destroy,
            width=80,
            height=30,
            fg_color="transparent",
            hover_color=("gray75", "gray25")
        ).pack(side="right", padx=10)
        
        # Main container with proper scrolling
        main_container = ctk.CTkFrame(memory_window)
        main_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Create scrollable frame with proper mouse wheel binding
        main_scroll = ctk.CTkScrollableFrame(
            main_container,
            width=710,
            height=480,
            scrollbar_button_color=("gray55", "gray45"),
            scrollbar_button_hover_color=("gray40", "gray60")
        )
        main_scroll.pack(fill="both", expand=True)
        
        # Enable mouse wheel scrolling on the frame and all children
        def _on_mousewheel(event):
            # Scroll the canvas directly
            if hasattr(main_scroll, '_parent_canvas'):
                main_scroll._parent_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel to the scrollable frame
        main_scroll.bind_all("<MouseWheel>", _on_mousewheel)
        main_scroll.bind_all("<Button-4>", lambda e: _on_mousewheel(type('', (), {'delta': 120})()))
        main_scroll.bind_all("<Button-5>", lambda e: _on_mousewheel(type('', (), {'delta': -120})()))
        
        # Status section
        status_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        status_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            status_frame,
            text="System Status",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=5)
        
        status_content = ctk.CTkFrame(status_frame)
        status_content.pack(fill="x", padx=20, pady=5)
        
        if hasattr(self, 'voice_memory') and self.voice_memory:
            status_text = "Voice Memory is Active"
            status_color = "green"
            status_icon = "✓"
        else:
            status_text = "Voice Memory is Inactive"
            status_color = "red"
            status_icon = "✗"
        
        ctk.CTkLabel(
            status_content,
            text=f"{status_icon} {status_text}",
            text_color=status_color,
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w")
        
        # Quick Actions section
        actions_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        actions_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            actions_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=5)
        
        # Action buttons in horizontal layout
        button_container = ctk.CTkFrame(actions_frame)
        button_container.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkButton(
            button_container,
            text="Export Data",
            command=lambda: self._quick_export_memory(),
            width=140,
            height=32
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_container,
            text="View Stats",
            command=lambda: self._quick_show_memory_stats(),
            width=140,
            height=32
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_container,
            text="Settings",
            command=lambda: [memory_window.destroy(), self.open_settings()],
            width=140,
            height=32
        ).pack(side="left", padx=5)
        
        # Conversations section
        conv_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        conv_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            conv_frame,
            text="Recent Voice Conversations",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=5)
        
        logging.info("Loading conversation history")
        
        # Load and display conversations in a contained frame
        conv_container = ctk.CTkFrame(conv_frame)
        conv_container.pack(fill="both", expand=True, padx=20, pady=5)
        
        self._load_conversation_history(conv_container)
        
        logging.info("Finished loading conversation history")
        
        # Unbind mousewheel when window is destroyed
        def on_close():
            main_scroll.unbind_all("<MouseWheel>")
            main_scroll.unbind_all("<Button-4>")
            main_scroll.unbind_all("<Button-5>")
            memory_window.destroy()
        
        memory_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def _quick_export_memory(self):
        """Quick export memory function"""
        if not hasattr(self, 'voice_memory') or not self.voice_memory:
            self.show_notification("Voice memory not available")
            return
        
        try:
            # Create export directory
            export_dir = os.path.expanduser("~/Documents/WhisperTranscriptions")
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(export_dir, f"voice_memory_export_{timestamp}.json")
            
            # Build export data manually
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "conversations": [],
                "statistics": {}
            }
            
            # Get conversations from database
            if hasattr(self.voice_memory, 'conversation_memory'):
                try:
                    recent = self.voice_memory.conversation_memory.get_recent(limit=100, session_only=False)
                    export_data["conversations"] = recent
                except Exception as conv_error:
                    logging.error(f"Failed to export conversations: {conv_error}")
            
            # Get conversations from context memory
            if hasattr(self.voice_memory, 'context_memory'):
                try:
                    # Use the conversation_history directly if available
                    if hasattr(self.voice_memory.context_memory, 'conversation_history'):
                        export_data["context_memory"] = {
                            "conversations": self.voice_memory.context_memory.conversation_history,
                            "preferences": getattr(self.voice_memory.context_memory, 'user_preferences', {}),
                            "patterns": getattr(self.voice_memory.context_memory, 'voice_command_patterns', {})
                        }
                except Exception as ctx_error:
                    logging.debug(f"Context memory export not critical: {ctx_error}")
            
            # Add analytics if available
            try:
                analytics = self.voice_memory.get_comprehensive_analytics(days=7)
                export_data["analytics"] = analytics
            except:
                pass
            
            # Save the export
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Get file size for feedback
            file_size = os.path.getsize(filename) / 1024  # Size in KB
            conv_count = len(export_data.get("conversations", []))
            
            # Show success dialog with details
            success_msg = (
                f"Memory Export Successful!\n\n"
                f"File: {os.path.basename(filename)}\n"
                f"Size: {file_size:.1f} KB\n"
                f"Conversations: {conv_count}\n"
                f"Location: ~/Documents/WhisperTranscriptions/"
            )
            
            # Create a simple info dialog
            import tkinter.messagebox as messagebox
            messagebox.showinfo("Export Complete", success_msg)
            
            # Also show notification
            self.show_notification(f"Exported {conv_count} conversations ({file_size:.1f}KB)")
            logging.info(f"Exported memory to {filename} ({file_size:.1f}KB, {conv_count} conversations)")
            
        except Exception as e:
            logging.error(f"Export failed: {e}")
            self.show_notification(f"Export failed: {e}")
    
    def _quick_show_memory_stats(self):
        """Quick show memory stats function"""
        if not hasattr(self, 'voice_memory') or not self.voice_memory:
            self.show_notification("Voice memory not available")
            return
        
        try:
            # Get comprehensive analytics
            analytics = self.voice_memory.get_comprehensive_analytics(days=7)
            
            # Extract key metrics
            voice_summary = analytics.get("voice_interaction_summary", {})
            total_conversations = voice_summary.get('total_conversations', 0)
            avg_confidence = voice_summary.get('average_confidence', 0)
            
            conv_patterns = analytics.get("conversation_patterns", {})
            voice_rate = conv_patterns.get('voice_interaction_rate', 0)
            
            # Create a stats window instead of just notification
            stats_window = ctk.CTkToplevel(self)
            stats_window.title("Voice Memory Statistics")
            stats_window.geometry("500x400")
            stats_window.transient(self)
            
            # Center the window
            stats_window.update_idletasks()
            x = (stats_window.winfo_screenwidth() // 2) - 250
            y = (stats_window.winfo_screenheight() // 2) - 200
            stats_window.geometry(f"500x400+{x}+{y}")
            
            # Title
            ctk.CTkLabel(
                stats_window,
                text="Voice Memory Statistics",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=10)
            
            # Create scrollable frame for stats
            stats_frame = ctk.CTkScrollableFrame(stats_window, width=460, height=300)
            stats_frame.pack(padx=20, pady=10, fill="both", expand=True)
            
            # Display stats sections
            sections = [
                ("Overview", [
                    ("Total Conversations", total_conversations),
                    ("Average Confidence", f"{avg_confidence:.1%}"),
                    ("Voice Interaction Rate", f"{voice_rate:.1%}")
                ]),
                ("Quality Metrics", [
                    ("Low Confidence Rate", f"{voice_summary.get('low_confidence_rate', 0):.1%}"),
                    ("Wake Word Rate", f"{voice_summary.get('wake_word_rate', 0):.1%}"),
                    ("Avg Audio Duration", f"{voice_summary.get('average_audio_duration', 0):.1f}s")
                ]),
                ("Recent Activity", [
                    ("Last Hour", conv_patterns.get('conversation_frequency', {}).get('last_hour', 0)),
                    ("Last Day", conv_patterns.get('conversation_frequency', {}).get('last_day', 0)),
                    ("Last Week", conv_patterns.get('conversation_frequency', {}).get('last_week', 0))
                ])
            ]
            
            for section_title, metrics in sections:
                # Section header
                section_frame = ctk.CTkFrame(stats_frame)
                section_frame.pack(fill="x", padx=10, pady=5)
                
                ctk.CTkLabel(
                    section_frame,
                    text=section_title,
                    font=ctk.CTkFont(size=14, weight="bold")
                ).pack(anchor="w", padx=10, pady=5)
                
                # Metrics
                for label, value in metrics:
                    metric_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
                    metric_frame.pack(fill="x", padx=20, pady=2)
                    
                    ctk.CTkLabel(
                        metric_frame,
                        text=f"{label}:",
                        font=ctk.CTkFont(size=12)
                    ).pack(side="left", padx=5)
                    
                    ctk.CTkLabel(
                        metric_frame,
                        text=str(value),
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="lightblue"
                    ).pack(side="left", padx=5)
            
            # Close button
            ctk.CTkButton(
                stats_window,
                text="Close",
                command=stats_window.destroy,
                width=100
            ).pack(pady=10)
            
            # Also show brief notification
            self.show_notification(f"Stats: {total_conversations} conversations, {avg_confidence:.0%} avg confidence")
            
            # Log detailed stats
            logging.info(f"Voice Memory Analytics displayed: {total_conversations} conversations")
            
        except Exception as e:
            logging.error(f"Stats failed: {e}")
            import traceback
            traceback.print_exc()
            self.show_notification(f"Stats failed: {e}")
    
    def _load_conversation_history(self, parent_frame, page=1, items_per_page=10):
        """Load and display conversation history with pagination"""
        logging.info(f"Loading conversation history page {page}")
        
        if not hasattr(self, 'voice_memory') or not self.voice_memory:
            logging.error("Voice memory not available")
            ctk.CTkLabel(
                parent_frame,
                text="Voice memory not available",
                text_color="red"
            ).pack(anchor="w", padx=10, pady=5)
            return
        
        # Clear existing content
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        try:
            # Get ALL conversations first for proper pagination
            all_conversations = []
            
            # Get from database (SQLite) - most recent first
            try:
                recent = self.voice_memory.conversation_memory.get_recent(limit=100, session_only=False)
                for conv in recent:
                    confidence = conv.get('transcription_confidence', 0.0) or 0.0
                    all_conversations.append({
                        'id': conv.get('id'),
                        'timestamp': conv.get('timestamp'),
                        'user_input': conv.get('user', ''),
                        'assistant_response': conv.get('assistant', ''),
                        'confidence': confidence,
                        'source': 'database'
                    })
            except Exception as e:
                logging.error(f"Failed to load from database: {e}")
            
            # Sort by timestamp (newest first)
            all_conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            if not all_conversations:
                ctk.CTkLabel(
                    parent_frame,
                    text="No conversations found yet\nStart recording to build your conversation history!",
                    text_color="gray",
                    font=ctk.CTkFont(size=11)
                ).pack(anchor="w", padx=10, pady=20)
                return
            
            # Calculate pagination
            total_items = len(all_conversations)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            
            # Add pagination controls at top
            pagination_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
            pagination_frame.pack(fill="x", padx=10, pady=5)
            
            # Page info
            ctk.CTkLabel(
                pagination_frame,
                text=f"Page {page} of {total_pages} ({total_items} total conversations)",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(side="left", padx=5)
            
            # Navigation buttons
            nav_frame = ctk.CTkFrame(pagination_frame, fg_color="transparent")
            nav_frame.pack(side="right", padx=5)
            
            if page > 1:
                ctk.CTkButton(
                    nav_frame,
                    text="← Previous",
                    command=lambda: self._load_conversation_history(parent_frame, page-1, items_per_page),
                    width=80,
                    height=28
                ).pack(side="left", padx=2)
            
            if page < total_pages:
                ctk.CTkButton(
                    nav_frame,
                    text="Next →",
                    command=lambda: self._load_conversation_history(parent_frame, page+1, items_per_page),
                    width=80,
                    height=28
                ).pack(side="left", padx=2)
            
            # Clear all button
            ctk.CTkButton(
                nav_frame,
                text="Clear All",
                command=lambda: self._clear_all_conversations(parent_frame),
                width=80,
                height=28,
                fg_color="red",
                hover_color="darkred"
            ).pack(side="left", padx=10)
            
            # Display conversations for current page
            conversations_to_show = all_conversations[start_idx:end_idx]
            
            logging.info(f"Displaying {len(conversations_to_show)} conversations on page {page}")
            
            for i, conv in enumerate(conversations_to_show):
                try:
                    # Add conversation ID for deletion capability
                    conv['display_index'] = start_idx + i + 1
                    self._create_conversation_widget_with_delete(parent_frame, conv, i)
                except Exception as e:
                    logging.error(f"Failed to create widget: {e}")
            
            # Add page navigation at bottom too if many items
            if total_pages > 1:
                bottom_nav = ctk.CTkFrame(parent_frame, fg_color="transparent")
                bottom_nav.pack(fill="x", padx=10, pady=10)
                
                # Quick page jump
                for p in range(1, min(total_pages + 1, 6)):  # Show first 5 pages
                    btn_color = "green" if p == page else "gray"
                    ctk.CTkButton(
                        bottom_nav,
                        text=str(p),
                        command=lambda pg=p: self._load_conversation_history(parent_frame, pg, items_per_page),
                        width=30,
                        height=28,
                        fg_color=btn_color
                    ).pack(side="left", padx=2)
                
                if total_pages > 5:
                    ctk.CTkLabel(bottom_nav, text="...").pack(side="left", padx=5)
                    ctk.CTkButton(
                        bottom_nav,
                        text=str(total_pages),
                        command=lambda: self._load_conversation_history(parent_frame, total_pages, items_per_page),
                        width=40,
                        height=28
                    ).pack(side="left", padx=2)
                    
        except Exception as e:
            logging.error(f"Failed to load conversation history: {e}")
            ctk.CTkLabel(
                parent_frame,
                text=f"Error loading conversations: {str(e)}",
                text_color="red"
            ).pack(anchor="w", padx=10, pady=5)
    
    def _create_conversation_widget_with_delete(self, parent, conversation, index):
        """Create a widget for a single conversation with delete button"""
        # Container frame
        conv_frame = ctk.CTkFrame(parent)
        conv_frame.pack(fill="x", padx=5, pady=3)
        
        # Parse data
        conv_id = conversation.get('id')
        display_idx = conversation.get('display_index', index + 1)
        timestamp = conversation.get('timestamp', 'Unknown')
        user_input = conversation.get('user_input', 'N/A')
        ai_response = conversation.get('assistant_response', 'N/A')
        confidence = conversation.get('confidence', 0.0)
        source = conversation.get('source', 'unknown')
        
        # Format timestamp
        try:
            if isinstance(timestamp, str) and 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%m/%d %H:%M:%S")
            else:
                time_str = str(timestamp)
        except:
            time_str = str(timestamp)
        
        # Header with delete button
        header_frame = ctk.CTkFrame(conv_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=5, pady=2)
        
        # Header text
        source_icon = {"database": "[DB]", "context": "[JSON]", "history": "[HIST]"}.get(source, "[?]")
        header_text = f"{source_icon} #{display_idx} - {time_str}"
        if confidence > 0:
            header_text += f" | Conf: {confidence:.1%}"
        
        ctk.CTkLabel(
            header_frame,
            text=header_text,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=5)
        
        # Delete button
        if conv_id and source == 'database':
            ctk.CTkButton(
                header_frame,
                text="Delete",
                command=lambda: self._delete_conversation(conv_id, conv_frame),
                width=50,
                height=20,
                font=ctk.CTkFont(size=10),
                fg_color="transparent",
                text_color="red",
                hover_color=("gray75", "gray25")
            ).pack(side="right", padx=5)
        
        # User input (simplified - just label for performance)
        if user_input and user_input != 'N/A':
            user_text = f"User: {user_input[:100]}{'...' if len(user_input) > 100 else ''}"
            ctk.CTkLabel(
                conv_frame,
                text=user_text,
                font=ctk.CTkFont(size=11),
                anchor="w",
                justify="left",
                text_color="lightblue",
                wraplength=600
            ).pack(anchor="w", padx=15, pady=2, fill="x")
        
        # AI response (simplified)
        if ai_response and ai_response != 'N/A':
            ai_text = f"AI: {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}"
            ctk.CTkLabel(
                conv_frame,
                text=ai_text,
                font=ctk.CTkFont(size=11),
                anchor="w",
                justify="left",
                text_color="lightgreen",
                wraplength=600
            ).pack(anchor="w", padx=15, pady=2, fill="x")
    
    def _delete_conversation(self, conv_id, widget_frame):
        """Delete a single conversation"""
        try:
            if hasattr(self.voice_memory.conversation_memory, 'delete_conversation'):
                self.voice_memory.conversation_memory.delete_conversation(conv_id)
            else:
                # Direct database deletion if method doesn't exist
                import sqlite3
                conn = sqlite3.connect(self.voice_memory.conversation_memory.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
                conn.commit()
                conn.close()
            
            # Remove widget
            widget_frame.destroy()
            self.show_notification("Conversation deleted")
            logging.info(f"Deleted conversation {conv_id}")
            
        except Exception as e:
            logging.error(f"Failed to delete conversation: {e}")
            self.show_notification(f"Delete failed: {e}")
    
    def _clear_all_conversations(self, parent_frame):
        """Clear all conversations with confirmation"""
        # Create confirmation dialog
        if messagebox.askyesno("Clear All Conversations", 
                               "Are you sure you want to delete ALL conversations?\n\nThis cannot be undone!"):
            try:
                # Clear database
                import sqlite3
                conn = sqlite3.connect(self.voice_memory.conversation_memory.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM conversations")
                conn.commit()
                conn.close()
                
                # Clear JSON memory
                if hasattr(self.voice_memory.context_memory, 'conversation_history'):
                    self.voice_memory.context_memory.conversation_history.clear()
                    self.voice_memory.context_memory.save_memory()
                
                self.show_notification("All conversations cleared")
                logging.info("Cleared all conversations")
                
                # Reload the view
                self._load_conversation_history(parent_frame)
                
            except Exception as e:
                logging.error(f"Failed to clear conversations: {e}")
                self.show_notification(f"Clear failed: {e}")
    
    def _create_conversation_widget(self, parent, conversation, index):
        """Create a widget for a single conversation entry"""
        # Container frame for this conversation
        conv_frame = ctk.CTkFrame(parent)
        conv_frame.pack(fill="x", padx=5, pady=3)
        
        # Parse conversation data
        timestamp = conversation.get('timestamp', 'Unknown')
        user_input = conversation.get('user_input', 'N/A')
        ai_response = conversation.get('assistant_response', 'N/A')
        confidence = conversation.get('confidence', 0.0)
        source = conversation.get('source', 'unknown')
        
        # Format timestamp
        try:
            if isinstance(timestamp, str) and 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%m/%d %H:%M:%S")
            else:
                time_str = str(timestamp)
        except:
            time_str = str(timestamp)
        
        # Header with timestamp and confidence
        header_frame = ctk.CTkFrame(conv_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=5, pady=2)
        
        # Create header text with source indicator (ASCII for Pi compatibility)
        source_icon = {"database": "[DB]", "context": "[JSON]", "history": "[HIST]"}.get(source, "[UNK]")
        header_text = f"{source_icon} #{index+1} - {time_str}"
        if confidence > 0:
            header_text += f" | Confidence: {confidence:.2%}"
        
        ctk.CTkLabel(
            header_frame,
            text=header_text,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", padx=5, pady=1)
        
        # User input - use textbox for better text wrapping
        if user_input and user_input != 'N/A':
            user_frame = ctk.CTkFrame(conv_frame, fg_color="transparent")
            user_frame.pack(fill="x", padx=15, pady=2)
            
            ctk.CTkLabel(
                user_frame,
                text="User:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="lightblue"
            ).pack(anchor="w", side="left", padx=(0, 5))
            
            # Truncate long text and use CTkTextbox for proper wrapping
            display_text = user_input[:200] + ('...' if len(user_input) > 200 else '')
            user_textbox = ctk.CTkTextbox(
                user_frame,
                height=40,
                width=550,
                font=ctk.CTkFont(size=11),
                fg_color="transparent",
                wrap="word"
            )
            user_textbox.pack(anchor="w", side="left", fill="x", expand=True)
            user_textbox.insert("1.0", display_text)
            user_textbox.configure(state="disabled")  # Make read-only
        
        # AI response (if available)
        if ai_response and ai_response != 'N/A':
            ai_frame = ctk.CTkFrame(conv_frame, fg_color="transparent")
            ai_frame.pack(fill="x", padx=15, pady=2)
            
            ctk.CTkLabel(
                ai_frame,
                text="AI:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="lightgreen"
            ).pack(anchor="w", side="left", padx=(0, 5))
            
            # Truncate long text and use CTkTextbox for proper wrapping
            display_text = ai_response[:200] + ('...' if len(ai_response) > 200 else '')
            ai_textbox = ctk.CTkTextbox(
                ai_frame,
                height=40,
                width=550,
                font=ctk.CTkFont(size=11),
                fg_color="transparent",
                wrap="word"
            )
            ai_textbox.pack(anchor="w", side="left", fill="x", expand=True)
            ai_textbox.insert("1.0", display_text)
            ai_textbox.configure(state="disabled")  # Make read-only
    
    def _calculate_confidence(self, result: dict) -> float:
        """Calculate average confidence from Whisper result"""
        try:
            if 'segments' in result and result['segments']:
                confidences = []
                for segment in result['segments']:
                    if 'avg_logprob' in segment:
                        # Convert log probability to confidence (rough approximation)
                        confidence = min(1.0, max(0.0, (segment['avg_logprob'] + 1.0)))
                        confidences.append(confidence)
                
                if confidences:
                    return sum(confidences) / len(confidences)
            
            # Fallback to default confidence
            return 0.8
            
        except Exception as e:
            logging.debug(f"Confidence calculation error: {e}")
            return 0.8
    
    def _enhance_prompt_with_memory(self, user_input: str) -> str:
        """Enhance user input with contextual information from voice memory"""
        if not self.voice_memory:
            return user_input
        
        try:
            # Get contextual hints from memory
            hints = self.voice_memory.get_contextual_response_hints(user_input)
            
            # Get recent conversation context
            fused_context = self.voice_memory.get_fused_context(
                context_window_size=3,
                voice_only=True,
                include_audio_metadata=False
            )
            
            # Build enhanced prompt
            enhanced_parts = []
            
            # Add conversation context if available
            if fused_context.get('conversation_history'):
                recent_convs = fused_context['conversation_history'][-2:]  # Last 2 conversations
                if recent_convs:
                    enhanced_parts.append("Previous conversation context:")
                    for conv in recent_convs:
                        user_text = conv.get('user_input', '')
                        ai_text = conv.get('assistant_response', '')
                        if user_text and ai_text:
                            enhanced_parts.append(f"User: {user_text[:100]}")
                            enhanced_parts.append(f"Assistant: {ai_text[:100]}")
                    enhanced_parts.append("")
            
            # Add similar past interactions
            similar = hints.get('similar_past_interactions', [])
            if similar:
                enhanced_parts.append("Similar past interactions:")
                for interaction in similar[:2]:  # Top 2 similar
                    past_user = interaction.get('user_input', '')
                    past_ai = interaction.get('assistant_response', '')
                    if past_user and past_ai:
                        enhanced_parts.append(f"Past: {past_user[:80]} -> {past_ai[:80]}")
                enhanced_parts.append("")
            
            # Add current user input
            enhanced_parts.append("Current user input:")
            enhanced_parts.append(user_input)
            
            # Add response guidance based on patterns
            patterns = hints.get('suggested_response_patterns', [])
            if patterns:
                enhanced_parts.append("")
                enhanced_parts.append("Response guidance:")
                for pattern in patterns[:2]:  # Top 2 patterns
                    suggestion = pattern.get('suggestion', '')
                    if suggestion:
                        enhanced_parts.append(f"- {suggestion}")
            
            enhanced_prompt = "\n".join(enhanced_parts)
            
            # Log the enhancement (debug only)
            if len(enhanced_prompt) > len(user_input):
                logging.debug(f"Enhanced prompt with {len(enhanced_parts)} context elements")
            
            return enhanced_prompt
            
        except Exception as e:
            logging.error(f"Failed to enhance prompt with memory: {e}")
            return user_input
    
    def copy_to_clipboard(self, text):
        """Copy text to system clipboard"""
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(['pbcopy'], input=text.encode(), check=True)
            elif system == "Windows":
                subprocess.run(['clip'], input=text.encode(), check=True)
            else:
                subprocess.run(['xclip', '-selection', 'clipboard'], 
                             input=text.encode(), check=True, timeout=1)
        except:
            pass
    
    def copy_last(self):
        """Copy last transcription"""
        if self.transcription_history:
            self.copy_to_clipboard(self.transcription_history[-1])
            self.show_notification("Copied last transcription")
    
    def copy_all(self):
        """Copy all transcriptions"""
        text = self.text_display.get("1.0", "end-1c")
        if text:
            self.copy_to_clipboard(text)
            self.show_notification("Copied all transcriptions")
    
    def export_transcription(self):
        """Export transcriptions to file"""
        # Create export directory in Documents
        export_dir = os.path.expanduser("~/Documents/WhisperTranscriptions")
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(export_dir, f"transcription_{timestamp}.txt")
        
        try:
            with open(filename, 'w') as f:
                f.write(self.text_display.get("1.0", "end-1c"))
            self.show_notification(f"Exported to {os.path.basename(filename)}")
            
            # Also show full path in terminal for reference
            print(f"Transcription saved to: {filename}")
        except Exception as e:
            self.show_notification(f"Export failed: {e}")
    
    def search_transcription(self):
        """Search in transcriptions (placeholder for search dialog)"""
        self.show_notification("Search feature coming soon!")
    
    def clear_transcriptions(self):
        """Clear all transcriptions"""
        self.text_display.delete("1.0", "end")
        self.ai_display.delete("1.0", "end")
        self.transcription_history.clear()
        self.update_status("Cleared", "green")
    
    def send_to_ai(self):
        """Send last transcription to AI"""
        if not self.transcription_history:
            self.show_notification("No transcription to send", "warning")
            return
        
        # Check if AI is enabled
        if not self.settings.settings.get("ai_enabled", False):
            self.show_notification("AI is disabled. Enable in settings.", "warning")
            return
        
        # Get current AI provider
        provider = self.get_current_ai_provider()
        if not provider:
            self.show_notification("AI provider not available. Check settings.", "error")
            return
        
        # For local AI, check if server is running and start if needed
        provider_type = self.settings.settings.get("ai_provider", "local")
        if provider_type == "local":
            if not self.local_ai.is_server_running():
                self.show_notification("Starting AI server...", "info")
                selected_model = self.settings.settings.get("ai_model", None)
                if not self.local_ai.start_server(selected_model):
                    self.show_notification("Failed to start AI server", "error")
                    return
        else:
            # For API providers, check availability
            if not provider.is_available():
                self.show_notification(f"AI provider not available: {provider.get_status()}", "error")
                return
        
        # Validate that we have transcription history
        if not self.transcription_history:
            self.show_notification("No transcription to send", "warning")
            return
            
        last_transcription = self.transcription_history[-1]
        
        # Input validation - sanitize and validate the text
        if not last_transcription or not last_transcription.strip():
            self.show_notification("Empty transcription - nothing to send", "warning")
            return
        
        # Get contextual hints from voice memory
        enhanced_prompt = self._enhance_prompt_with_memory(last_transcription)
        
        # Limit text length to prevent excessive API usage
        max_length = 4000  # Reasonable limit for most AI providers
        if len(enhanced_prompt) > max_length:
            enhanced_prompt = enhanced_prompt[:max_length] + "..."
            logging.info(f"Truncated long enhanced prompt from {len(enhanced_prompt)} to {max_length} chars")
        
        self.show_notification("Sending to AI...", "info")
        
        # Process in background thread
        def process_ai():
            try:
                if provider_type == "local":
                    # Use existing send_to_ai method for local AI with enhanced prompt
                    result = self.local_ai.send_to_ai(enhanced_prompt)
                else:
                    # Use send_message for API providers with enhanced prompt
                    response = provider.send_message(enhanced_prompt)
                    if response.startswith("Error:"):
                        result = {"success": False, "error": response}
                    else:
                        result = {"success": True, "response": response}
                
                # Update UI in main thread (fix thread safety by capturing values)
                def handle_response(res=result, txt=last_transcription):
                    self._handle_ai_response(res, txt)
                self.after(0, handle_response)
                
            except Exception as e:
                error_result = {"success": False, "error": f"Unexpected error: {str(e)}"}
                def handle_error(res=error_result, txt=last_transcription):
                    self._handle_ai_response(res, txt)
                self.after(0, handle_error)
        
        # Import at top of method to avoid repeated imports
        import threading
        thread = threading.Thread(target=process_ai, daemon=True)
        thread.start()
    
    def auto_send_to_ai(self, text):
        """Automatically send transcription to AI if enabled"""
        # Check if AI auto-send is enabled
        if not self.settings.settings.get("ai_auto_send", False):
            return
        
        # Check if AI is enabled
        if not self.settings.settings.get("ai_enabled", False):
            return
        
        # Input validation - sanitize and validate the text
        if not text or not text.strip():
            logging.debug("Skipping auto-send of empty transcription")
            return
        
        # Limit text length to prevent excessive API usage
        max_length = 4000  # Reasonable limit for most AI providers
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logging.info(f"Auto-send: Truncated long transcription to {max_length} chars")
        
        # Get current AI provider
        provider = self.get_current_ai_provider()
        if not provider:
            logging.warning("AI provider not available for auto-send")
            return
        
        # For local AI, check if server is running and start if needed
        provider_type = self.settings.settings.get("ai_provider", "local")
        if provider_type == "local":
            if not self.local_ai.is_server_running():
                selected_model = self.settings.settings.get("ai_model", None)
                if not self.local_ai.start_server(selected_model):
                    return
        else:
            # For API providers, check availability
            if not provider.is_available():
                logging.warning(f"AI provider not available for auto-send: {provider.get_status()}")
                return
        
        # Process in background thread
        def process_ai():
            try:
                if provider_type == "local":
                    # Use existing send_to_ai method for local AI
                    result = self.local_ai.send_to_ai(text)
                else:
                    # Use send_message for API providers
                    response = provider.send_message(text)
                    if response.startswith("Error:"):
                        result = {"success": False, "error": response}
                    else:
                        result = {"success": True, "response": response}
                
                # Update UI in main thread (fix thread safety by capturing values)
                def handle_response(res=result, txt=text):
                    self._handle_ai_response(res, txt)
                self.after(0, handle_response)
                
            except Exception as e:
                error_result = {"success": False, "error": f"Unexpected error: {str(e)}"}
                def handle_error(res=error_result, txt=text):
                    self._handle_ai_response(res, txt)
                self.after(0, handle_error)
        
        # Import at top of method to avoid repeated imports
        import threading
        thread = threading.Thread(target=process_ai, daemon=True)
        thread.start()
    
    def _handle_ai_response(self, result, original_text):
        """Handle AI response in main thread"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if result["success"]:
            ai_text = f"[{timestamp}] User: {original_text[:50]}...\nAI: {result['response']}\n\n"
            self.ai_display.insert("end", ai_text)
            self.ai_display.see("end")
            self.show_notification("AI responded", "success")
            
            # Store AI interaction in voice memory
            self._store_ai_interaction(original_text, result['response'])
        else:
            error_text = f"[{timestamp}] Error: {result['error']}\n\n"
            self.ai_display.insert("end", error_text)
            self.ai_display.see("end")
            self.show_notification(f"AI error: {result['error'][:30]}", "error")
    
    def open_settings(self):
        """Open settings window"""
        if not hasattr(self, 'settings_window') or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.settings)
        else:
            self.settings_window.lift()
    
    def update_status(self, text, color="white"):
        """Update status label"""
        self.ui_queue.put({
            'type': 'status',
            'text': text,
            'color': color
        })
    
    def update_ai_panel_visibility(self):
        """Show or hide AI panel based on settings"""
        if self.local_ai.is_enabled():
            # Show AI panel with proper grid weights
            self.ai_frame.grid(row=0, column=1, padx=(5, 0), pady=0, sticky="nsew")
            self.content_frame.grid_columnconfigure(0, weight=2)  # Transcription panel
            self.content_frame.grid_columnconfigure(1, weight=1)  # AI panel
        else:
            # Hide AI panel and expand transcription panel
            self.ai_frame.grid_forget()
            self.content_frame.grid_columnconfigure(0, weight=1)  # Full width for transcription
            self.content_frame.grid_columnconfigure(1, weight=0)  # No AI panel
        
        # Update AI model label
        self.update_ai_model_label()
    
    def update_ai_model_label(self):
        """Update the AI model label with current provider and status"""
        provider_type = self.settings.settings.get("ai_provider", "local")
        ai_enabled = self.settings.settings.get("ai_enabled", False)
        
        if not ai_enabled:
            self.ai_model_label.configure(text="AI: Disabled", text_color="gray")
            return
        
        # Get current provider
        provider = self.get_current_ai_provider()
        
        if provider_type == "local":
            # Local AI status
            current_model = self.local_ai.get_current_model()
            if current_model:
                self.ai_model_label.configure(text=f"AI: {current_model}", text_color="green")
            elif self.local_ai.is_server_running():
                self.ai_model_label.configure(text="AI: Starting...", text_color="orange")
            else:
                self.ai_model_label.configure(text="AI: Local (Not Running)", text_color="orange")
        
        elif provider_type == "claude":
            # Claude API status
            if provider and provider.is_available():
                self.ai_model_label.configure(text="AI: Claude API", text_color="green")
            else:
                self.ai_model_label.configure(text="AI: Claude (No API Key)", text_color="red")
        
        elif provider_type == "openai":
            # OpenAI API status
            if provider and provider.is_available():
                model = self.settings.settings.get("openai_model", "gpt-3.5-turbo")
                model_short = model.replace("gpt-", "GPT-")
                self.ai_model_label.configure(text=f"AI: {model_short}", text_color="green")
            else:
                self.ai_model_label.configure(text="AI: OpenAI (No API Key)", text_color="red")
        
        else:
            self.ai_model_label.configure(text="AI: Unknown", text_color="gray")
    
    def check_ai_status(self):
        """Periodically check and update AI server status"""
        self.update_ai_model_label()
        # Check every 5 seconds
        self.after(5000, self.check_ai_status)
    
    def get_current_ai_provider(self) -> Optional[AIProvider]:
        """Get the currently selected AI provider instance"""
        provider_type = self.settings.settings.get("ai_provider", "local")
        
        try:
            if provider_type == "local":
                return self.local_ai
            elif provider_type == "claude":
                api_key = self.settings.settings.get("claude_api_key", "")
                if not api_key:
                    logging.warning("Claude API key not configured")
                    return None
                return ClaudeAPIProvider(api_key)
            elif provider_type == "openai":
                api_key = self.settings.settings.get("openai_api_key", "")
                model = self.settings.settings.get("openai_model", "gpt-3.5-turbo")
                if not api_key:
                    logging.warning("OpenAI API key not configured")
                    return None
                return OpenAIProvider(api_key, model)
            else:
                logging.error(f"Unknown AI provider: {provider_type}")
                return None
        except Exception as e:
            logging.error(f"Failed to create AI provider {provider_type}: {e}")
            return None
    
    def show_notification(self, message, notification_type="info"):
        """Show temporary notification"""
        color_map = {
            "info": "blue",
            "success": "green",
            "warning": "orange",
            "error": "red"
        }
        color = color_map.get(notification_type, "yellow")
        self.update_status(message, color)
        if notification_type != "error":
            self.after(3000, lambda: self.update_status("* Ready", "green"))
    
    def process_ui_queue(self):
        """Process UI updates from queue"""
        try:
            while True:
                update = self.ui_queue.get_nowait()
                
                if update['type'] == 'transcription':
                    self.text_display.insert("end", update['text'])
                    self.text_display.see("end")
                elif update['type'] == 'status':
                    self.status_label.configure(
                        text=update['text'],
                        text_color=update.get('color', 'white')
                    )
                elif update['type'] == 'info':
                    self.text_display.insert("end", update['text'])
                elif update['type'] == 'model_update':
                    self.model_label.configure(text=f"Model: {update['model']}")
                    
        except queue.Empty:
            pass
        
        self.after(50, self.process_ui_queue)

class SettingsWindow(ctk.CTkToplevel):
    """Settings window with comprehensive options"""
    
    def __init__(self, parent, settings):
        super().__init__(parent)
        
        self.parent = parent
        self.settings = settings
        self.test_expanded = False  # Initialize test panel state
        
        self.title("Settings")
        
        # Make window smaller and fit screen better
        screen_height = self.winfo_screenheight()
        window_height = min(500, int(screen_height * 0.7))  # Max 70% of screen height
        window_width = 650
        
        self.geometry(f"{window_width}x{window_height}")
        self.minsize(600, 400)
        self.maxsize(800, int(screen_height * 0.9))
        
        # Allow resizing
        self.resizable(True, True)
        
        # Center the window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.winfo_screenheight() // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Get available audio devices
        self.audio_devices = self.get_audio_devices()
        
        self.create_settings_ui()
    
    def _bind_mousewheel(self, widget):
        """Bind mouse wheel events to scrollable widget"""
        # Bind mouse wheel events for scrolling
        def _on_mousewheel(event):
            widget._parent_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            widget._parent_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            # Linux support
            widget._parent_canvas.bind_all("<Button-4>", lambda e: widget._parent_canvas.yview_scroll(-1, "units"))
            widget._parent_canvas.bind_all("<Button-5>", lambda e: widget._parent_canvas.yview_scroll(1, "units"))
        
        def _unbind_from_mousewheel(event):
            widget._parent_canvas.unbind_all("<MouseWheel>")
            widget._parent_canvas.unbind_all("<Button-4>")
            widget._parent_canvas.unbind_all("<Button-5>")
        
        widget.bind('<Enter>', _bind_to_mousewheel)
        widget.bind('<Leave>', _unbind_from_mousewheel)
    
    def get_audio_devices(self):
        """Get list of available audio input devices"""
        devices = []
        try:
            import sounddevice as sd
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    devices.append(f"{i}: {device['name']}")
        except:
            devices = ["Default"]
        return devices if devices else ["Default"]
    
    def create_settings_ui(self):
        """Create settings interface"""
        
        # FIRST: Create bottom button frame (must be created first to reserve space)
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(side="bottom", fill="x", padx=15, pady=10)
        
        # Add buttons to the frame
        self._create_bottom_buttons(button_frame)
        
        # THEN: Create main content area that will fill remaining space
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        
        # Title
        title_label = ctk.CTkLabel(
            main_container, 
            text="Settings",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(5, 10))
        
        # Tabs for different settings categories
        self.tabview = ctk.CTkTabview(main_container)
        self.tabview.pack(fill="both", expand=True, padx=5)
        
        # Create tabs
        self.tabview.add("Audio")
        self.tabview.add("Transcription")
        self.tabview.add("Interface")
        self.tabview.add("AI Integration")
        self.tabview.add("Voice Memory")
        self.tabview.add("Advanced")
        
        # Audio Settings Tab
        audio_tab = self.tabview.tab("Audio")
        audio_frame = ctk.CTkScrollableFrame(audio_tab, height=250)
        audio_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Enable mouse wheel scrolling
        self._bind_mousewheel(audio_frame)
        
        ctk.CTkLabel(
            audio_frame, 
            text="Audio Configuration", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Microphone selection with actual devices
        mic_frame = ctk.CTkFrame(audio_frame)
        mic_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            mic_frame, 
            text="Select Microphone:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        self.mic_combo = ctk.CTkComboBox(
            mic_frame,
            values=self.audio_devices,
            width=400,
            height=35
        )
        self.mic_combo.pack(padx=10, pady=5)
        
        # Set current device
        if self.settings.settings.get("audio_device"):
            self.mic_combo.set(self.settings.settings["audio_device"])
        else:
            self.mic_combo.set(self.audio_devices[0] if self.audio_devices else "Default")
        
        # Test microphone button
        self.test_button = ctk.CTkButton(
            mic_frame,
            text=" Test Microphone",
            command=self.toggle_test_microphone,
            width=200,
            height=35
        )
        self.test_button.pack(pady=10)
        
        # Test results frame (initially hidden)
        self.test_frame = ctk.CTkFrame(mic_frame)
        # Don't pack initially - will be shown when test starts
        
        # Test status label
        self.test_status = ctk.CTkLabel(
            self.test_frame,
            text="Ready to test microphone",
            font=ctk.CTkFont(size=11)
        )
        self.test_status.pack(pady=5)
        
        # Audio level bar
        level_container = ctk.CTkFrame(self.test_frame)
        level_container.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            level_container,
            text="Level:",
            font=ctk.CTkFont(size=10)
        ).pack(side="left", padx=5)
        
        self.test_level_bar = ctk.CTkProgressBar(level_container, width=150)
        self.test_level_bar.pack(side="left", padx=5)
        self.test_level_bar.set(0)
        
        self.test_level_text = ctk.CTkLabel(
            level_container,
            text="0%",
            font=ctk.CTkFont(size=10)
        )
        self.test_level_text.pack(side="left", padx=5)
        
        # Mini waveform canvas
        from tkinter import Canvas
        self.test_waveform = Canvas(
            self.test_frame,
            height=60,
            bg="#212121" if self.settings.settings.get("theme", "dark") == "dark" else "#f0f0f0",
            highlightthickness=0
        )
        self.test_waveform.pack(fill="x", padx=20, pady=5)
        
        # Test controls
        test_controls = ctk.CTkFrame(self.test_frame)
        test_controls.pack(pady=5)
        
        self.test_record_btn = ctk.CTkButton(
            test_controls,
            text="Record 2s",
            command=lambda: self.start_test_recording(),
            width=80,
            height=28
        )
        self.test_record_btn.pack(side="left", padx=5)
        
        self.test_play_btn = ctk.CTkButton(
            test_controls,
            text="Playback",
            command=lambda: self.playback_test_recording(),
            width=80,
            height=28,
            state="disabled"
        )
        self.test_play_btn.pack(side="left", padx=5)
        
        # Storage for test recording
        self.test_audio_data = None
        self.test_expanded = False
        
        # Audio processing options
        process_frame = ctk.CTkFrame(audio_frame)
        process_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            process_frame,
            text="Audio Processing:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        # Noise reduction
        self.noise_var = ctk.BooleanVar(value=self.settings.settings["noise_reduction"])
        noise_check = ctk.CTkCheckBox(
            process_frame,
            text="Enable Noise Reduction (Reduces background noise)",
            variable=self.noise_var,
            font=ctk.CTkFont(size=12)
        )
        noise_check.pack(anchor="w", padx=20, pady=5)
        
        # VAD
        self.vad_var = ctk.BooleanVar(value=self.settings.settings["vad_enabled"])
        vad_check = ctk.CTkCheckBox(
            process_frame,
            text="Voice Activity Detection (Auto start/stop recording)",
            variable=self.vad_var,
            font=ctk.CTkFont(size=12)
        )
        vad_check.pack(anchor="w", padx=20, pady=5)
        
        # Sample rate selection
        rate_frame = ctk.CTkFrame(audio_frame)
        rate_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            rate_frame,
            text="Sample Rate:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        self.rate_combo = ctk.CTkComboBox(
            rate_frame,
            values=["16000 Hz", "44100 Hz", "48000 Hz"],
            width=200
        )
        self.rate_combo.pack(anchor="w", padx=20, pady=5)
        self.rate_combo.set(f"{self.settings.settings.get('sample_rate', 44100)} Hz")
        
        # Transcription Settings Tab
        trans_tab = self.tabview.tab("Transcription")
        trans_frame = ctk.CTkScrollableFrame(trans_tab, height=250)
        trans_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Enable mouse wheel scrolling
        self._bind_mousewheel(trans_frame)
        
        ctk.CTkLabel(
            trans_frame, 
            text="Transcription Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Model selection
        model_frame = ctk.CTkFrame(trans_frame)
        model_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            model_frame,
            text="Whisper Model:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(
            model_frame,
            text="Larger models are more accurate but slower",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", padx=10)
        
        self.model_combo = ctk.CTkComboBox(
            model_frame,
            values=["tiny (39MB, fastest)", "base (74MB)", "small (244MB)", "medium (769MB, most accurate)"],
            width=400,
            height=35
        )
        self.model_combo.pack(padx=10, pady=5)
        
        # Set current model
        model_map = {
            "tiny": "tiny (39MB, fastest)",
            "base": "base (74MB)",
            "small": "small (244MB)",
            "medium": "medium (769MB, most accurate)"
        }
        current_model = self.settings.settings.get("model", "tiny")
        self.model_combo.set(model_map.get(current_model, "tiny (39MB, fastest)"))
        
        # Language selection
        lang_frame = ctk.CTkFrame(trans_frame)
        lang_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            lang_frame,
            text="Language:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        languages = [
            "en (English)", "es (Spanish)", "fr (French)", "de (German)",
            "it (Italian)", "pt (Portuguese)", "ru (Russian)", "ja (Japanese)",
            "ko (Korean)", "zh (Chinese)", "ar (Arabic)", "hi (Hindi)"
        ]
        
        self.lang_combo = ctk.CTkComboBox(
            lang_frame,
            values=languages,
            width=400,
            height=35
        )
        self.lang_combo.pack(padx=10, pady=5)
        
        # Set current language
        current_lang = self.settings.settings.get("language", "en")
        for lang_option in languages:
            if lang_option.startswith(current_lang):
                self.lang_combo.set(lang_option)
                break
        
        # Hailo Integration
        if self.parent.hailo.hailo_available:
            hailo_frame = ctk.CTkFrame(trans_frame)
            hailo_frame.pack(fill="x", pady=10)
            
            ctk.CTkLabel(
                hailo_frame,
                text="Hailo AI Integration:",
                font=ctk.CTkFont(size=12)
            ).pack(anchor="w", padx=10, pady=5)
            
            self.hailo_var = ctk.BooleanVar(value=self.settings.settings.get("hailo_integration", False))
            hailo_check = ctk.CTkCheckBox(
                hailo_frame,
                text="Enable Hailo AI Audio Enhancement",
                variable=self.hailo_var,
                font=ctk.CTkFont(size=12)
            )
            hailo_check.pack(anchor="w", padx=20, pady=5)
            
            # Add explanation
            ctk.CTkLabel(
                hailo_frame,
                text=" AI-powered audio processing:\n   - Noise suppression\n   - Volume normalization\n   - Speech clarity enhancement",
                font=ctk.CTkFont(size=10),
                text_color="gray",
                justify="left"
            ).pack(anchor="w", padx=40, pady=(0, 5))
            
            ctk.CTkLabel(
                hailo_frame,
                text="[OK] Hailo AI processor detected - Audio enhancement available",
                font=ctk.CTkFont(size=10),
                text_color="green"
            ).pack(anchor="w", padx=20)
        
        # Interface Settings Tab
        ui_tab = self.tabview.tab("Interface")
        ui_frame = ctk.CTkScrollableFrame(ui_tab, height=250)
        ui_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Enable mouse wheel scrolling
        self._bind_mousewheel(ui_frame)
        
        ctk.CTkLabel(
            ui_frame,
            text="Interface Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Theme selection
        theme_frame = ctk.CTkFrame(ui_frame)
        theme_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            theme_frame,
            text="Application Theme:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        self.theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["Dark Mode", "Light Mode", "System Default"],
            width=400,
            height=35,
            command=self.preview_theme
        )
        self.theme_combo.pack(padx=10, pady=5)
        
        # Set current theme
        theme_map = {"dark": "Dark Mode", "light": "Light Mode", "system": "System Default"}
        self.theme_combo.set(theme_map.get(self.settings.settings.get("theme", "dark"), "Dark Mode"))
        
        # Window configuration
        window_frame = ctk.CTkFrame(ui_frame)
        window_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            window_frame,
            text="Window Configuration:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        # Window size
        ctk.CTkLabel(
            window_frame,
            text="Window Size:",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=20, pady=2)
        
        self.size_combo = ctk.CTkComboBox(
            window_frame,
            values=["Compact (400x300)", "Standard (600x500)", "Large (800x600)"],
            width=300,
            height=30
        )
        self.size_combo.pack(anchor="w", padx=30, pady=5)
        
        # Set current size
        size_map = {
            "compact": "Compact (400x300)",
            "standard": "Standard (600x500)",
            "large": "Large (800x600)"
        }
        self.size_combo.set(size_map.get(self.settings.settings.get("window_size", "standard"), "Standard (600x500)"))
        
        # Always on top
        self.top_var = ctk.BooleanVar(value=self.settings.settings.get("always_on_top", True))
        top_check = ctk.CTkCheckBox(
            window_frame,
            text="Keep Window Always on Top",
            variable=self.top_var,
            font=ctk.CTkFont(size=12)
        )
        top_check.pack(anchor="w", padx=20, pady=5)
        
        # Text settings
        text_frame = ctk.CTkFrame(ui_frame)
        text_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            text_frame,
            text="Text Display:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        # Font size
        self.font_label = ctk.CTkLabel(
            text_frame,
            text=f"Font Size: {self.settings.settings.get('font_size', 12)}pt",
            font=ctk.CTkFont(size=11)
        )
        self.font_label.pack(anchor="w", padx=20, pady=5)
        
        self.font_slider = ctk.CTkSlider(
            text_frame,
            from_=10,
            to=20,
            number_of_steps=10,
            width=300,
            command=self.update_font_label
        )
        self.font_slider.pack(anchor="w", padx=30, pady=5)
        self.font_slider.set(self.settings.settings.get("font_size", 12))
        
        # AI Integration Settings Tab
        ai_tab = self.tabview.tab("AI Integration")
        ai_frame = ctk.CTkScrollableFrame(ai_tab, height=250)
        ai_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Enable mouse wheel scrolling
        self._bind_mousewheel(ai_frame)
        
        ctk.CTkLabel(
            ai_frame,
            text="AI Integration Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Provider selection comes FIRST
        provider_frame = ctk.CTkFrame(ai_frame)
        provider_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            provider_frame,
            text="1. AI Provider Selection:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(
            provider_frame,
            text="Choose your AI provider",
            font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=20, pady=2)
        
        # Initialize provider variable
        self.ai_provider_var = ctk.StringVar(value=self.settings.settings.get("ai_provider", "local"))
        self.claude_api_key_var = ctk.StringVar(value=self.settings.settings.get("claude_api_key", ""))
        self.openai_api_key_var = ctk.StringVar(value=self.settings.settings.get("openai_api_key", ""))
        self.openai_model_var = ctk.StringVar(value=self.settings.settings.get("openai_model", "gpt-3.5-turbo"))
        
        # Provider radio buttons
        provider_radio_frame = ctk.CTkFrame(provider_frame)
        provider_radio_frame.pack(padx=10, pady=5, fill="x")
        
        self.local_radio = ctk.CTkRadioButton(
            provider_radio_frame,
            text="Local AI (Run models on this device)",
            variable=self.ai_provider_var,
            value="local",
            command=self.on_provider_change
        )
        self.local_radio.pack(anchor="w", padx=10, pady=3)
        
        self.claude_radio = ctk.CTkRadioButton(
            provider_radio_frame,
            text="Claude API (Anthropic's cloud service)",
            variable=self.ai_provider_var,
            value="claude",
            command=self.on_provider_change
        )
        self.claude_radio.pack(anchor="w", padx=10, pady=3)
        
        self.openai_radio = ctk.CTkRadioButton(
            provider_radio_frame,
            text="OpenAI API (GPT models)",
            variable=self.ai_provider_var,
            value="openai",
            command=self.on_provider_change
        )
        self.openai_radio.pack(anchor="w", padx=10, pady=3)
        
        # Claude API key input
        self.claude_frame = ctk.CTkFrame(provider_frame)
        self.claude_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            self.claude_frame,
            text="Claude API Key:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5,0))
        
        self.claude_api_entry = ctk.CTkEntry(
            self.claude_frame,
            textvariable=self.claude_api_key_var,
            placeholder_text="Enter your Claude API key...",
            show="*",
            width=400
        )
        self.claude_api_entry.pack(padx=10, pady=(2,10))
        # Save API key when it changes
        self.claude_api_key_var.trace("w", lambda *args: self.save_api_keys())
        
        # OpenAI configuration
        self.openai_frame = ctk.CTkFrame(provider_frame)
        self.openai_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            self.openai_frame,
            text="OpenAI API Key:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5,0))
        
        self.openai_api_entry = ctk.CTkEntry(
            self.openai_frame,
            textvariable=self.openai_api_key_var,
            placeholder_text="Enter your OpenAI API key...",
            show="*",
            width=400
        )
        self.openai_api_entry.pack(padx=10, pady=(2,5))
        # Save API key when it changes
        self.openai_api_key_var.trace("w", lambda *args: self.save_api_keys())
        
        ctk.CTkLabel(
            self.openai_frame,
            text="OpenAI Model:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5,0))
        
        self.openai_model_combo = ctk.CTkComboBox(
            self.openai_frame,
            values=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"],
            variable=self.openai_model_var,
            width=400
        )
        self.openai_model_combo.pack(padx=10, pady=(2,10))
        # Save model selection when it changes
        self.openai_model_var.trace("w", lambda *args: self.save_api_keys())
        
        # Model selection comes SECOND - select model before starting server
        self.model_frame = ctk.CTkFrame(ai_frame)
        self.model_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            self.model_frame,
            text="2. Local AI Model Selection:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(
            self.model_frame,
            text="Select which local AI model to use (only for Local AI provider)",
            font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=20, pady=2)
        
        # Check if TinyLlama is available
        if not self.parent.local_ai.is_tinyllama_available():
            # Show download option
            download_frame = ctk.CTkFrame(self.model_frame)
            download_frame.pack(padx=10, pady=5)
            
            ctk.CTkLabel(
                download_frame,
                text="TinyLlama model not found locally",
                font=ctk.CTkFont(size=11),
                text_color="orange"
            ).pack(side="left", padx=5)
            
            self.download_btn = ctk.CTkButton(
                download_frame,
                text="Download TinyLlama (638 MB)",
                command=self.download_tinyllama,
                width=200
            )
            self.download_btn.pack(side="left", padx=5)
        
        # Get available models from the LLM server
        try:
            available_models = self.get_available_ai_models()
        except:
            available_models = []
        
        # Always include TinyLlama in the list (even if not downloaded yet)
        if "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" not in available_models:
            available_models.insert(0, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        
        self.ai_model_combo = ctk.CTkComboBox(
            self.model_frame,
            values=available_models if available_models else ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"],
            width=400,
            height=35
        )
        self.ai_model_combo.pack(padx=10, pady=5)
        
        # Set current model
        current_model = self.settings.settings.get("ai_model", available_models[0] if available_models else "")
        if current_model in available_models:
            self.ai_model_combo.set(current_model)
        else:
            self.ai_model_combo.set(available_models[0] if available_models else "")
        
        # Server control comes SECOND - start server with selected model
        self.server_frame = ctk.CTkFrame(ai_frame)
        self.server_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            self.server_frame,
            text="3. Local Server Control:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(
            self.server_frame,
            text="Start the local AI server with the selected model (only for Local AI provider)",
            font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=20, pady=2)
        
        server_buttons_frame = ctk.CTkFrame(self.server_frame)
        server_buttons_frame.pack(padx=10, pady=5)
        
        self.start_server_btn = ctk.CTkButton(
            server_buttons_frame,
            text="Start AI Server",
            command=self.start_ai_server,
            width=120
        )
        self.start_server_btn.pack(side="left", padx=5)
        
        self.stop_server_btn = ctk.CTkButton(
            server_buttons_frame,
            text="Stop AI Server",
            command=self.stop_ai_server,
            width=120
        )
        self.stop_server_btn.pack(side="left", padx=5)
        
        self.server_status_label = ctk.CTkLabel(
            self.server_frame,
            text="Server Status: Not Running",
            font=ctk.CTkFont(size=11)
        )
        self.server_status_label.pack(padx=10, pady=5)
        
        # AI Enable/Disable comes THIRD - only enable after server is running
        ai_enable_frame = ctk.CTkFrame(ai_frame)
        ai_enable_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            ai_enable_frame,
            text="4. Enable AI Features:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(
            ai_enable_frame,
            text="Enable these features after configuring your AI provider",
            font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=20, pady=2)
        
        self.ai_enabled_var = ctk.BooleanVar(value=self.settings.settings.get("ai_enabled", False))
        self.ai_enabled_checkbox = ctk.CTkCheckBox(
            ai_enable_frame,
            text="Enable AI Processing (shows AI panel)",
            variable=self.ai_enabled_var,
            command=self.on_ai_settings_change
        )
        self.ai_enabled_checkbox.pack(anchor="w", padx=20, pady=5)
        
        # Auto-send setting
        self.ai_auto_send_var = ctk.BooleanVar(value=self.settings.settings.get("ai_auto_send", False))
        self.ai_auto_send_checkbox = ctk.CTkCheckBox(
            ai_enable_frame,
            text="Automatically send transcriptions to AI",
            variable=self.ai_auto_send_var,
            command=self.on_ai_settings_change
        )
        self.ai_auto_send_checkbox.pack(anchor="w", padx=20, pady=5)
        
        # Set initial provider visibility
        self.on_provider_change()
        
        # Update server status
        self.update_ai_server_status()
        
        # Voice Memory Settings Tab
        memory_tab = self.tabview.tab("Voice Memory")
        memory_frame = ctk.CTkScrollableFrame(memory_tab, height=250)
        memory_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Enable mouse wheel scrolling
        self._bind_mousewheel(memory_frame)
        
        ctk.CTkLabel(
            memory_frame,
            text="Voice Memory Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Voice Memory Status
        status_frame = ctk.CTkFrame(memory_frame)
        status_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            status_frame,
            text="Memory System Status:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Status indicator
        memory_available = VOICE_MEMORY_AVAILABLE and hasattr(self.parent, 'voice_memory') and self.parent.voice_memory is not None
        status_text = "✅ Available and Active" if memory_available else "❌ Not Available"
        status_color = "green" if memory_available else "red"
        
        self.memory_status_label = ctk.CTkLabel(
            status_frame,
            text=status_text,
            text_color=status_color,
            font=ctk.CTkFont(size=11)
        )
        self.memory_status_label.pack(anchor="w", padx=20, pady=2)
        
        # Enable/Disable Voice Memory
        ctk.CTkLabel(
            memory_frame,
            text="Core Settings:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(15, 5))
        
        self.voice_memory_enabled = ctk.CTkCheckBox(
            memory_frame,
            text="Enable Voice Memory System",
            command=self.on_voice_memory_change
        )
        self.voice_memory_enabled.pack(anchor="w", padx=20, pady=5)
        
        # Audio Metadata
        self.voice_memory_audio_metadata = ctk.CTkCheckBox(
            memory_frame,
            text="Store Audio Metadata (confidence, duration, etc.)",
            command=self.on_voice_memory_change
        )
        self.voice_memory_audio_metadata.pack(anchor="w", padx=20, pady=5)
        
        # Pattern Learning
        self.voice_memory_pattern_learning = ctk.CTkCheckBox(
            memory_frame,
            text="Enable Pattern Learning and Analysis",
            command=self.on_voice_memory_change
        )
        self.voice_memory_pattern_learning.pack(anchor="w", padx=20, pady=5)
        
        # Context Window
        context_frame = ctk.CTkFrame(memory_frame)
        context_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            context_frame,
            text="Context Window Size:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(
            context_frame,
            text="Number of recent interactions to include in AI context",
            font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=20, pady=2)
        
        self.context_limit_var = ctk.StringVar(value=str(self.settings.settings.get("voice_memory_context_limit", 10)))
        self.context_limit_label = ctk.CTkLabel(context_frame, text=f"Context Limit: {self.context_limit_var.get()}")
        self.context_limit_label.pack(anchor="w", padx=20, pady=2)
        
        self.context_limit_slider = ctk.CTkSlider(
            context_frame,
            from_=3,
            to=20,
            number_of_steps=17,
            command=self.update_context_limit_label
        )
        self.context_limit_slider.pack(anchor="w", padx=30, pady=5)
        self.context_limit_slider.set(self.settings.settings.get("voice_memory_context_limit", 10))
        
        # Memory Management
        mgmt_frame = ctk.CTkFrame(memory_frame)
        mgmt_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            mgmt_frame,
            text="Memory Management:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Auto-save
        self.voice_memory_auto_save = ctk.CTkCheckBox(
            mgmt_frame,
            text="Auto-save memory data",
            command=self.on_voice_memory_change
        )
        self.voice_memory_auto_save.pack(anchor="w", padx=20, pady=5)
        
        # Compression
        self.voice_memory_compression = ctk.CTkCheckBox(
            mgmt_frame,
            text="Enable memory compression",
            command=self.on_voice_memory_change
        )
        self.voice_memory_compression.pack(anchor="w", padx=20, pady=5)
        
        # Memory Actions
        actions_frame = ctk.CTkFrame(memory_frame)
        actions_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            actions_frame,
            text="Memory Actions:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Buttons row
        button_row = ctk.CTkFrame(actions_frame)
        button_row.pack(fill="x", padx=20, pady=5)
        
        self.memory_export_btn = ctk.CTkButton(
            button_row,
            text="Export Memory",
            command=self.export_voice_memory,
            width=120
        )
        self.memory_export_btn.pack(side="left", padx=5)
        
        self.memory_clear_btn = ctk.CTkButton(
            button_row,
            text="Clear Memory",
            command=self.clear_voice_memory,
            width=120,
            fg_color="red",
            hover_color="darkred"
        )
        self.memory_clear_btn.pack(side="left", padx=5)
        
        self.memory_stats_btn = ctk.CTkButton(
            button_row,
            text="View Stats",
            command=self.show_memory_stats,
            width=120
        )
        self.memory_stats_btn.pack(side="left", padx=5)
        
        # Advanced Settings Tab
        adv_tab = self.tabview.tab("Advanced")
        adv_frame = ctk.CTkScrollableFrame(adv_tab, height=250)
        adv_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Enable mouse wheel scrolling
        self._bind_mousewheel(adv_frame)
        
        ctk.CTkLabel(
            adv_frame,
            text="Advanced Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Performance options
        perf_frame = ctk.CTkFrame(adv_frame)
        perf_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            perf_frame,
            text="Performance Options:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        # GPU acceleration (placeholder)
        self.gpu_var = ctk.BooleanVar(value=False)
        gpu_check = ctk.CTkCheckBox(
            perf_frame,
            text="Enable GPU Acceleration (if available)",
            variable=self.gpu_var,
            font=ctk.CTkFont(size=12),
            state="disabled"  # Disabled for now
        )
        gpu_check.pack(anchor="w", padx=20, pady=5)
        
        # Debug options
        debug_frame = ctk.CTkFrame(adv_frame)
        debug_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            debug_frame,
            text="Debug Options:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=10, pady=5)
        
        self.debug_var = ctk.BooleanVar(value=self.settings.settings.get("debug_logging", False))
        debug_check = ctk.CTkCheckBox(
            debug_frame,
            text="Enable Debug Logging",
            variable=self.debug_var,
            font=ctk.CTkFont(size=12)
        )
        debug_check.pack(anchor="w", padx=20, pady=5)
        
        ctk.CTkButton(
            debug_frame,
            text="View Log File",
            command=self.view_log,
            width=150,
            height=30
        ).pack(anchor="w", padx=30, pady=5)
        
        # Load initial values for voice memory settings
        self.load_voice_memory_settings()
        
    def load_voice_memory_settings(self):
        """Load initial values for voice memory settings"""
        # Set checkbox values
        self.voice_memory_enabled.select() if self.settings.settings.get("voice_memory_enabled", True) else self.voice_memory_enabled.deselect()
        self.voice_memory_audio_metadata.select() if self.settings.settings.get("voice_memory_audio_metadata", True) else self.voice_memory_audio_metadata.deselect()
        self.voice_memory_pattern_learning.select() if self.settings.settings.get("voice_memory_pattern_learning", True) else self.voice_memory_pattern_learning.deselect()
        self.voice_memory_auto_save.select() if self.settings.settings.get("voice_memory_auto_save", True) else self.voice_memory_auto_save.deselect()
        self.voice_memory_compression.select() if self.settings.settings.get("voice_memory_compression", True) else self.voice_memory_compression.deselect()
        
        # Update memory status
        self.update_memory_status()
    
    def update_memory_status(self):
        """Update the memory system status display"""
        memory_available = VOICE_MEMORY_AVAILABLE and hasattr(self.parent, 'voice_memory') and self.parent.voice_memory is not None
        if memory_available:
            self.memory_status_label.configure(text="✅ Available and Active", text_color="green")
        else:
            self.memory_status_label.configure(text="❌ Not Available", text_color="red")
    
    def update_context_limit_label(self, value):
        """Update context limit label"""
        self.context_limit_label.configure(text=f"Context Limit: {int(float(value))}")
    
    def on_voice_memory_change(self):
        """Handle voice memory setting changes"""
        # Update settings immediately
        self.settings.settings["voice_memory_enabled"] = self.voice_memory_enabled.get()
        self.settings.settings["voice_memory_audio_metadata"] = self.voice_memory_audio_metadata.get()
        self.settings.settings["voice_memory_pattern_learning"] = self.voice_memory_pattern_learning.get()
        self.settings.settings["voice_memory_auto_save"] = self.voice_memory_auto_save.get()
        self.settings.settings["voice_memory_compression"] = self.voice_memory_compression.get()
        self.settings.settings["voice_memory_context_limit"] = int(self.context_limit_slider.get())
    
    def export_voice_memory(self):
        """Export voice memory data"""
        if not hasattr(self.parent, 'voice_memory') or not self.parent.voice_memory:
            self.show_notification("Voice memory not available", "error")
            return
        
        try:
            # Create export directory
            export_dir = os.path.expanduser("~/Documents/WhisperTranscriptions")
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(export_dir, f"voice_memory_export_{timestamp}.json")
            
            # Export session data
            export_data = self.parent.voice_memory.export_session_data(
                format="json",
                include_analytics=True
            )
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.show_notification(f"Memory exported to {os.path.basename(filename)}", "success")
            
        except Exception as e:
            self.show_notification(f"Export failed: {e}", "error")
    
    def clear_voice_memory(self):
        """Clear voice memory with confirmation"""
        if not hasattr(self.parent, 'voice_memory') or not self.parent.voice_memory:
            self.show_notification("Voice memory not available", "error")
            return
        
        # Simple confirmation using CTk
        import tkinter.messagebox as msgbox
        
        if msgbox.askyesno("Clear Memory", "Are you sure you want to clear all voice memory? This cannot be undone."):
            try:
                result = self.parent.voice_memory.clear_memory(
                    clear_context=True,
                    clear_conversation=True,
                    keep_preferences=True
                )
                
                if result.get("context_cleared", False) or result.get("conversation_cleared", False):
                    self.show_notification("Memory cleared successfully", "success")
                else:
                    self.show_notification("Memory clear completed with warnings", "warning")
                    
            except Exception as e:
                self.show_notification(f"Clear failed: {e}", "error")
    
    def show_memory_stats(self):
        """Show voice memory statistics"""
        if not hasattr(self.parent, 'voice_memory') or not self.parent.voice_memory:
            self.show_notification("Voice memory not available", "error")
            return
        
        try:
            analytics = self.parent.voice_memory.get_comprehensive_analytics(days=7)
            
            # Create stats display window
            stats_window = ctk.CTkToplevel(self)
            stats_window.title("Voice Memory Statistics")
            stats_window.geometry("600x400")
            stats_window.transient(self)
            stats_window.grab_set()
            
            # Add scrollable text area
            stats_frame = ctk.CTkScrollableFrame(stats_window)
            stats_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Format and display stats
            stats_text = []
            stats_text.append("📊 Voice Memory Statistics (Last 7 Days)")
            stats_text.append("=" * 50)
            
            # Basic stats
            system_perf = analytics.get("system_performance", {})
            stats_text.append(f"Total Interactions: {system_perf.get('total_interactions', 0)}")
            stats_text.append(f"Successful Interactions: {system_perf.get('successful_interactions', 0)}")
            stats_text.append(f"Average Response Time: {system_perf.get('average_response_time', 0):.2f}s")
            stats_text.append("")
            
            # Memory stats
            memory_stats = analytics.get("memory_efficiency", {})
            stats_text.append("🧠 Memory System Status:")
            stats_text.append(f"Context Memory: {'Active' if memory_stats.get('context_memory_available') else 'Inactive'}")
            stats_text.append(f"Conversation Memory: {'Active' if memory_stats.get('conversation_memory_available') else 'Inactive'}")
            stats_text.append(f"Memory Fusion Calls: {memory_stats.get('memory_fusion_calls', 0)}")
            stats_text.append("")
            
            # Transcription quality
            quality = analytics.get("transcription_quality", {})
            if quality:
                summary = quality.get("summary", {})
                stats_text.append("🎤 Transcription Quality:")
                stats_text.append(f"Average Confidence: {summary.get('overall_avg_confidence', 0):.2f}")
                stats_text.append(f"Total Transcriptions: {summary.get('total_transcriptions', 0)}")
                stats_text.append("")
            
            # Recommendations
            recommendations = analytics.get("recommendations", [])
            if recommendations:
                stats_text.append("💡 Recommendations:")
                for rec in recommendations[:3]:
                    stats_text.append(f"• {rec.get('message', 'No message')}")
                stats_text.append("")
            
            # Display stats
            stats_label = ctk.CTkLabel(
                stats_frame,
                text="\n".join(stats_text),
                font=ctk.CTkFont(family="monospace", size=12),
                justify="left"
            )
            stats_label.pack(anchor="w", padx=10, pady=10)
            
        except Exception as e:
            self.show_notification(f"Stats failed: {e}", "error")
    
    def show_notification(self, message, type="info"):
        """Show a simple notification"""
        # Simple console output for now
        print(f"[{type.upper()}] {message}")
        
    def _create_bottom_buttons(self, button_frame):
        """Create the bottom button row"""
        button_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkButton(
            button_frame,
            text="Save Settings",
            command=self.save_settings,
            width=150,
            height=35,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        ).grid(row=0, column=0, padx=5, sticky="e")
        
        ctk.CTkButton(
            button_frame,
            text="Reset Defaults",
            command=self.reset_defaults,
            width=150,
            height=35,
            font=ctk.CTkFont(size=13)
        ).grid(row=0, column=1, padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy,
            width=150,
            height=35,
            font=ctk.CTkFont(size=13)
        ).grid(row=0, column=2, padx=5, sticky="w")
    
    def toggle_test_microphone(self):
        """Toggle the test microphone panel"""
        if not self.test_expanded:
            self.test_frame.pack(fill="x", padx=10, pady=5)
            self.test_button.configure(text=" Hide Test Panel")
            self.test_expanded = True
        else:
            self.test_frame.pack_forget()
            self.test_button.configure(text=" Test Microphone")
            self.test_expanded = False
            self.test_level_bar.set(0)
            self.test_level_text.configure(text="0%")
    
    def start_test_recording(self):
        """Start inline microphone test recording"""
        try:
            import sounddevice as sd
            import numpy as np
            import threading
            
            # Get device
            device_str = self.mic_combo.get()
            device_index = int(device_str.split(":")[0]) if ":" in device_str else None
            sample_rate = 44100
            duration = 2
            
            # Update UI
            self.test_status.configure(text="Recording for 2 seconds...")
            self.test_record_btn.configure(state="disabled")
            self.test_play_btn.configure(state="disabled")
            self.test_level_bar.set(0)
            
            def record():
                try:
                    # Record audio
                    self.test_audio_data = sd.rec(
                        int(duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        device=device_index,
                        dtype=np.float32
                    )
                    sd.wait()
                    
                    # Flatten and analyze
                    self.test_audio_data = self.test_audio_data.flatten()
                    avg_level = np.abs(self.test_audio_data).mean()
                    max_level = np.abs(self.test_audio_data).max()
                    
                    # Draw waveform
                    self.draw_test_waveform(self.test_audio_data)
                    
                    # Update status
                    if avg_level > 0.001:
                        self.test_status.configure(
                            text=f"Recording complete! Avg: {avg_level:.4f}, Peak: {max_level:.3f}"
                        )
                        self.test_play_btn.configure(state="normal")
                        self.test_level_bar.set(min(1.0, avg_level * 50))
                        self.test_level_text.configure(text=f"{int(min(100, avg_level * 5000))}%")
                    else:
                        self.test_status.configure(
                            text="WARNING: No audio detected. Check microphone."
                        )
                        self.test_level_bar.set(0)
                        self.test_level_text.configure(text="0%")
                    
                    self.test_record_btn.configure(state="normal")
                    
                except Exception as e:
                    self.test_status.configure(text=f"Error: {str(e)[:50]}")
                    self.test_record_btn.configure(state="normal")
            
            # Start recording in thread
            thread = threading.Thread(target=record, daemon=True)
            thread.start()
            
        except Exception as e:
            self.test_status.configure(text=f"Error: {str(e)[:50]}")
    
    def playback_test_recording(self):
        """Play back the test recording"""
        if self.test_audio_data is not None:
            try:
                import sounddevice as sd
                import threading
                
                self.test_status.configure(text="Playing recording...")
                self.test_play_btn.configure(state="disabled")
                
                def play():
                    sd.play(self.test_audio_data, 44100)
                    sd.wait()
                    self.test_status.configure(text="Playback complete")
                    self.test_play_btn.configure(state="normal")
                
                thread = threading.Thread(target=play, daemon=True)
                thread.start()
                
            except Exception as e:
                self.test_status.configure(text=f"Playback error: {str(e)[:30]}")
                self.test_play_btn.configure(state="normal")
    
    def draw_test_waveform(self, audio_data):
        """Draw waveform on mini canvas"""
        try:
            import numpy as np
            
            self.test_waveform.delete("all")
            
            # Get canvas dimensions
            canvas_width = self.test_waveform.winfo_width()
            canvas_height = 60  # Fixed height
            
            if canvas_width <= 1:
                canvas_width = 300  # Default width
            
            # Downsample for display
            samples_to_show = min(len(audio_data), 500)
            step = max(1, len(audio_data) // samples_to_show)
            downsampled = audio_data[::step]
            
            # Normalize
            max_val = np.max(np.abs(downsampled))
            if max_val > 0:
                normalized = downsampled / max_val
            else:
                normalized = downsampled
            
            # Draw waveform
            mid_y = canvas_height // 2
            x_step = canvas_width / len(normalized)
            
            # Draw center line
            self.test_waveform.create_line(
                0, mid_y, canvas_width, mid_y,
                fill="#444444" if self.settings.settings.get("theme", "dark") == "dark" else "#cccccc",
                dash=(3, 3)
            )
            
            # Draw waveform
            points = []
            for i, val in enumerate(normalized):
                x = i * x_step
                y = mid_y - (val * mid_y * 0.7)
                points.extend([x, y])
            
            if len(points) >= 4:
                self.test_waveform.create_line(
                    points,
                    fill="#00ff00" if self.settings.settings.get("theme", "dark") == "dark" else "#0080ff",
                    width=2
                )
        except Exception as e:
            print(f"Waveform draw error: {e}")
    
    def preview_theme(self, theme_name):
        """Live theme preview"""
        theme_map = {"Dark Mode": "dark", "Light Mode": "light", "System Default": "system"}
        theme = theme_map.get(theme_name, "dark")
        if theme != "system":
            ctk.set_appearance_mode(theme)
    
    
    def update_font_label(self, value):
        """Update font size label with slider value"""
        self.font_label.configure(text=f"Font Size: {int(value)}pt")
    
    def view_log(self):
        """Open log file viewer"""
        try:
            import subprocess
            subprocess.run(['xdg-open', '/tmp/whisper_pro.log'])
        except:
            self.parent.show_notification("Log file: /tmp/whisper_pro.log")
    
    def reset_defaults(self):
        """Reset all settings to defaults"""
        # Reset UI elements
        self.theme_combo.set("Dark Mode")
        self.model_combo.set("tiny (39MB, fastest)")
        self.lang_combo.set("en (English)")
        self.size_combo.set("Standard (600x500)")
        self.top_var.set(True)
        self.font_slider.set(12)
        self.noise_var.set(False)
        self.vad_var.set(False)
        if hasattr(self, 'hailo_var'):
            self.hailo_var.set(False)
        
        self.parent.show_notification("Settings reset to defaults")
    
    def save_settings(self):
        """Save all settings"""
        # Extract values from UI
        theme_map = {"Dark Mode": "dark", "Light Mode": "light", "System Default": "system"}
        size_map = {
            "Compact (400x300)": "compact",
            "Standard (600x500)": "standard",
            "Large (800x600)": "large"
        }
        
        # Extract model name
        model_text = self.model_combo.get()
        model = model_text.split()[0] if model_text else "tiny"
        
        # Extract language code
        lang_text = self.lang_combo.get()
        lang = lang_text.split()[0] if lang_text else "en"
        
        # Extract device
        device_text = self.mic_combo.get()
        
        # Extract sample rate
        rate_text = self.rate_combo.get()
        sample_rate = int(rate_text.split()[0]) if rate_text else 44100
        
        # Update settings
        self.settings.settings.update({
            "theme": theme_map.get(self.theme_combo.get(), "dark"),
            "model": model,
            "language": lang,
            "window_size": size_map.get(self.size_combo.get(), "standard"),
            "always_on_top": self.top_var.get(),
            "font_size": int(self.font_slider.get()),
            "noise_reduction": self.noise_var.get(),
            "vad_enabled": self.vad_var.get(),
            "audio_device": device_text,
            "sample_rate": sample_rate,
            "debug_logging": self.debug_var.get() if hasattr(self, 'debug_var') else False
        })
        
        if hasattr(self, 'hailo_var'):
            self.settings.settings["hailo_integration"] = self.hailo_var.get()
        
        # AI Integration settings
        if hasattr(self, 'ai_enabled_var'):
            self.settings.settings["ai_enabled"] = self.ai_enabled_var.get()
        if hasattr(self, 'ai_auto_send_var'):
            self.settings.settings["ai_auto_send"] = self.ai_auto_send_var.get()
        if hasattr(self, 'ai_model_combo'):
            self.settings.settings["ai_model"] = self.ai_model_combo.get()
        
        # AI Provider settings
        if hasattr(self, 'ai_provider_var'):
            self.settings.settings["ai_provider"] = self.ai_provider_var.get()
        if hasattr(self, 'claude_api_key_var'):
            self.settings.settings["claude_api_key"] = self.claude_api_key_var.get()
        if hasattr(self, 'openai_api_key_var'):
            self.settings.settings["openai_api_key"] = self.openai_api_key_var.get()
        if hasattr(self, 'openai_model_var'):
            self.settings.settings["openai_model"] = self.openai_model_var.get()
        
        # Save to file
        self.settings.save_settings()
        
        # Apply debug logging level
        if self.debug_var.get():
            logging.getLogger().setLevel(logging.DEBUG)
            logging.info("Debug logging enabled")
        else:
            logging.getLogger().setLevel(logging.INFO)
            logging.info("Debug logging disabled")
        
        # Apply changes
        self.parent.apply_theme()
        
        # Apply font size to text display
        new_font_size = int(self.font_slider.get())
        self.parent.text_display.configure(font=ctk.CTkFont(size=new_font_size))
        
        # Apply window size changes immediately
        size_map_reverse = {
            "Compact (400x300)": "400x300",
            "Standard (600x500)": "600x500", 
            "Large (800x600)": "800x600"
        }
        new_size = size_map_reverse.get(self.size_combo.get(), "600x500")
        self.parent.geometry(new_size)
        
        # Apply window changes
        if self.top_var.get():
            self.parent.attributes("-topmost", True)
        else:
            self.parent.attributes("-topmost", False)
        
        # Check if model changed and reload if needed
        current_model = getattr(self.parent.model, 'model_name', 'tiny') if self.parent.model else 'tiny'
        if model != current_model:
            self.parent.show_notification(f"Loading {model} model...")
            # Reload model in background thread
            def reload_model():
                try:
                    self.parent.model = whisper.load_model(model)
                    self.parent.ui_queue.put({
                        'type': 'status',
                        'text': '* Ready',
                        'color': 'green'
                    })
                    logging.info(f"Reloaded Whisper model: {model}")
                except Exception as e:
                    logging.error(f"Failed to reload model: {e}")
                    self.parent.ui_queue.put({
                        'type': 'status',
                        'text': 'Model reload failed',
                        'color': 'red'
                    })
            
            threading.Thread(target=reload_model, daemon=True).start()
            
            # Update model label in UI
            self.parent.model_label.configure(text=f"Model: {model}")
        
        # Update Hailo status in UI
        if hasattr(self.parent, 'hailo_label'):
            hailo_enabled = self.hailo_var.get() if hasattr(self, 'hailo_var') else False
            if self.parent.hailo.hailo_available and hailo_enabled:
                self.parent.hailo_label.configure(text="Hailo: ON", text_color="green")
            else:
                self.parent.hailo_label.configure(text="Hailo: OFF", text_color="gray")
        
        # Check if audio device changed
        if device_text and device_text != self.parent.settings.settings.get("audio_device"):
            try:
                import sounddevice as sd
                # Extract device index
                device_idx = int(device_text.split(":")[0])
                self.parent.device_index = device_idx
                # Update sample rate for new device
                devices = sd.query_devices()
                if device_idx < len(devices):
                    self.parent.device_sample_rate = int(devices[device_idx]['default_samplerate'])
                    logging.info(f"Switched to audio device: {device_text}")
            except Exception as e:
                logging.error(f"Failed to switch audio device: {e}")
        
        self.parent.show_notification("Settings applied successfully!")
        self.destroy()
    
    def get_available_ai_models(self):
        """Get list of available AI models"""
        from pathlib import Path
        models_dir = Path.home() / "simple-llm-server" / "models"
        if not models_dir.exists():
            return ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "phi-2.Q4_K_M.gguf"]
        
        models = []
        for model_file in models_dir.glob("*.gguf"):
            models.append(model_file.name)
        
        return sorted(models) if models else ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "phi-2.Q4_K_M.gguf"]
    
    def on_ai_settings_change(self):
        """Handle AI settings changes"""
        # Update parent's LocalAI instance immediately
        if hasattr(self.parent, 'local_ai'):
            self.parent.local_ai.settings.settings["ai_enabled"] = self.ai_enabled_var.get()
            self.parent.local_ai.settings.settings["ai_auto_send"] = self.ai_auto_send_var.get()
            self.parent.local_ai.settings.save_settings()
            # Update AI panel visibility
            self.parent.update_ai_panel_visibility()
    
    def save_api_keys(self):
        """Save API keys when they change"""
        if hasattr(self, 'claude_api_key_var'):
            self.settings.settings["claude_api_key"] = self.claude_api_key_var.get()
        if hasattr(self, 'openai_api_key_var'):
            self.settings.settings["openai_api_key"] = self.openai_api_key_var.get()
        if hasattr(self, 'openai_model_var'):
            self.settings.settings["openai_model"] = self.openai_model_var.get()
        self.settings.save_settings()
        
        # Update main UI to reflect changes
        if hasattr(self.parent, 'update_ai_model_label'):
            self.parent.update_ai_model_label()
    
    def on_provider_change(self):
        """Handle AI provider selection change"""
        provider = self.ai_provider_var.get()
        
        # Save the provider selection immediately
        self.settings.settings["ai_provider"] = provider
        self.settings.save_settings()
        
        # Update main UI to reflect provider change
        if hasattr(self.parent, 'update_ai_model_label'):
            self.parent.update_ai_model_label()
        
        # Show/hide frames based on provider selection
        if provider == "local":
            # Show local AI components
            if hasattr(self, 'model_frame'):
                self.model_frame.pack(fill="x", pady=10)
            if hasattr(self, 'server_frame'):
                self.server_frame.pack(fill="x", pady=10)
            # Hide API components
            if hasattr(self, 'claude_frame'):
                self.claude_frame.pack_forget()
            if hasattr(self, 'openai_frame'):
                self.openai_frame.pack_forget()
        
        elif provider == "claude":
            # Hide local AI components
            if hasattr(self, 'model_frame'):
                self.model_frame.pack_forget()
            if hasattr(self, 'server_frame'):
                self.server_frame.pack_forget()
            # Show Claude API components
            if hasattr(self, 'claude_frame'):
                self.claude_frame.pack(fill="x", padx=10, pady=5)
            # Hide OpenAI components
            if hasattr(self, 'openai_frame'):
                self.openai_frame.pack_forget()
        
        elif provider == "openai":
            # Hide local AI components
            if hasattr(self, 'model_frame'):
                self.model_frame.pack_forget()
            if hasattr(self, 'server_frame'):
                self.server_frame.pack_forget()
            # Hide Claude API components
            if hasattr(self, 'claude_frame'):
                self.claude_frame.pack_forget()
            # Show OpenAI API components
            if hasattr(self, 'openai_frame'):
                self.openai_frame.pack(fill="x", padx=10, pady=5)
    
    def start_ai_server(self):
        """Start the AI server"""
        try:
            model_name = self.ai_model_combo.get() if hasattr(self, 'ai_model_combo') else None
            success = self.parent.local_ai.start_server(model_name)
            if success:
                self.parent.show_notification(f"AI server started with {model_name}")
                self.update_ai_server_status()
            else:
                self.parent.show_notification("Failed to start AI server", "error")
        except Exception as e:
            self.parent.show_notification(f"Error starting AI server: {str(e)}", "error")
    
    def stop_ai_server(self):
        """Stop the AI server"""
        try:
            self.parent.local_ai.stop_server()
            self.parent.show_notification("AI server stopped")
            self.update_ai_server_status()
        except Exception as e:
            self.parent.show_notification(f"Error stopping AI server: {str(e)}", "error")
    
    def download_tinyllama(self):
        """Download TinyLlama model with progress dialog"""
        # Create progress dialog
        progress_dialog = ctk.CTkToplevel(self)
        progress_dialog.title("Downloading TinyLlama")
        progress_dialog.geometry("450x150")
        progress_dialog.transient(self)
        progress_dialog.grab_set()
        
        # Center the dialog
        progress_dialog.update_idletasks()
        x = (progress_dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (progress_dialog.winfo_screenheight() // 2) - (150 // 2)
        progress_dialog.geometry(f"450x150+{x}+{y}")
        
        # Progress widgets
        progress_label = ctk.CTkLabel(
            progress_dialog,
            text="Downloading TinyLlama model (638 MB)...",
            font=ctk.CTkFont(size=12)
        )
        progress_label.pack(pady=10)
        
        progress_bar = ctk.CTkProgressBar(progress_dialog, width=400)
        progress_bar.pack(pady=10)
        progress_bar.set(0)
        
        status_label = ctk.CTkLabel(
            progress_dialog,
            text="Starting download...",
            font=ctk.CTkFont(size=10)
        )
        status_label.pack(pady=5)
        
        # Download in background thread
        import threading
        download_success = False
        
        def download_thread():
            nonlocal download_success
            def update_progress(downloaded, total):
                if total > 0:
                    percent = (downloaded / total) * 100
                    mb_downloaded = downloaded / (1024*1024)
                    mb_total = total / (1024*1024)
                    
                    # Update UI in main thread
                    progress_dialog.after(0, lambda: progress_bar.set(percent / 100))
                    progress_dialog.after(0, lambda: status_label.configure(
                        text=f"{mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent:.0f}%)"
                    ))
            
            download_success = self.parent.local_ai.download_tinyllama(update_progress)
            
            # Close dialog and update UI
            progress_dialog.after(0, lambda: self.on_download_complete(progress_dialog, download_success))
        
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()
    
    def on_download_complete(self, dialog, success):
        """Handle download completion"""
        dialog.destroy()
        
        if success:
            self.parent.show_notification("TinyLlama downloaded successfully!")
            # Hide download button and refresh model list
            if hasattr(self, 'download_btn'):
                self.download_btn.pack_forget()
            # Refresh available models
            available_models = self.get_available_ai_models()
            self.ai_model_combo.configure(values=available_models)
            self.ai_model_combo.set("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        else:
            self.parent.show_notification("Failed to download TinyLlama", "error")
    
    def update_ai_server_status(self):
        """Update AI server status display"""
        if hasattr(self, 'server_status_label'):
            is_running = self.parent.local_ai.is_server_running()
            current_model = self.parent.local_ai.get_current_model()
            
            if is_running and current_model:
                status_text = f"Server Status: Running ({current_model})"
                status_color = "green"
            elif is_running:
                status_text = "Server Status: Running"
                status_color = "green"
            else:
                status_text = "Server Status: Not Running"
                status_color = "gray"
            
            self.server_status_label.configure(text=status_text, text_color=status_color)
            
            # Update main UI model label
            self.parent.update_ai_model_label()

def main():
    """Main entry point"""
    app = WhisperTranscribePro()
    app.mainloop()

if __name__ == "__main__":
    main()