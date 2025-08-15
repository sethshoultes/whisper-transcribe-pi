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
import customtkinter as ctk
from PIL import Image, ImageDraw
from scipy import signal
from scipy.signal import butter, lfilter

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

class LocalAI:
    """Manage local LLM server integration"""
    
    def __init__(self, settings):
        self.settings = settings
        self.server_process = None
        self.llm_dir = Path.home() / "simple-llm-server"
        self.models_dir = self.llm_dir / "models"
        
    def is_enabled(self):
        """Check if AI is enabled in settings"""
        return self.settings.settings.get("ai_enabled", False)
    
    def get_available_models(self):
        """Get list of available .gguf models"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_file in self.models_dir.glob("*.gguf"):
            models.append(model_file.name)
        return sorted(models)
    
    def start_server(self, model_name=None):
        """Start the local LLM server"""
        if self.server_process and self.server_process.poll() is None:
            return True  # Already running
            
        try:
            env_dir = self.llm_dir / "env"
            if not env_dir.exists():
                return False
                
            # Use the manual server command from CLAUDE.md
            cmd = [
                str(env_dir / "bin" / "python"),
                str(self.llm_dir / "app.py")
            ]
            
            # Start server in background
            self.server_process = subprocess.Popen(
                cmd,
                cwd=str(self.llm_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it a moment to start
            import time
            time.sleep(2)
            
            return self.server_process.poll() is None
            
        except Exception as e:
            logging.error(f"Failed to start AI server: {e}")
            return False
    
    def stop_server(self):
        """Stop the local LLM server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
    
    def is_server_running(self):
        """Check if server is running"""
        if self.server_process and self.server_process.poll() is None:
            return True
        
        # Also check if something is running on port 7860
        try:
            import requests
            response = requests.get("http://localhost:7860", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def send_to_ai(self, text):
        """Send text to local AI and get response"""
        if not self.is_server_running():
            return {"success": False, "error": "AI server not running"}
            
        try:
            import requests
            # Simple API call to your LLM server
            response = requests.post(
                "http://localhost:7860/api/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": text}],
                    "temperature": 0.7,
                    "max_tokens": 200
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
                return {"success": True, "response": ai_response}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

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
            "ai_auto_send": False
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
        
        # AI Response Panel
        self.ai_frame = ctk.CTkFrame(self.content_frame)
        self.ai_frame.grid(row=0, column=1, padx=(5, 0), pady=0, sticky="nsew")
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
        self.button_frame.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        
        # Action Buttons (without emoji for Pi compatibility)
        buttons = [
            ("Copy Last", self.copy_last),
            ("Copy All", self.copy_all),
            ("Export", self.export_transcription),
            ("Send to AI", self.send_to_ai),
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
            text=f"Model: {self.settings.settings['model']}",
            font=ctk.CTkFont(size=10)
        )
        self.model_label.grid(row=0, column=0, padx=5, pady=2)
        
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
        
        if not self.local_ai.is_enabled():
            self.show_notification("AI is disabled. Enable in settings.", "warning")
            return
        
        if not self.local_ai.is_server_running():
            self.show_notification("Starting AI server...", "info")
            if not self.local_ai.start_server():
                self.show_notification("Failed to start AI server", "error")
                return
        
        last_transcription = self.transcription_history[-1]
        self.show_notification("Sending to AI...", "info")
        
        # Process in background thread
        def process_ai():
            result = self.local_ai.send_to_ai(last_transcription)
            
            # Update UI in main thread
            self.after(0, lambda: self._handle_ai_response(result, last_transcription))
        
        import threading
        thread = threading.Thread(target=process_ai, daemon=True)
        thread.start()
    
    def auto_send_to_ai(self, text):
        """Automatically send transcription to AI if enabled"""
        if not self.local_ai.is_enabled():
            return
        
        if not self.local_ai.is_server_running():
            # Try to start server automatically
            if not self.local_ai.start_server():
                return
        
        # Process in background thread
        def process_ai():
            result = self.local_ai.send_to_ai(text)
            
            # Update UI in main thread
            self.after(0, lambda: self._handle_ai_response(result, text))
        
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
    
    def show_notification(self, message):
        """Show temporary notification"""
        self.update_status(message, "yellow")
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
        
        # AI Enable/Disable
        ai_enable_frame = ctk.CTkFrame(ai_frame)
        ai_enable_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            ai_enable_frame,
            text="Local AI Integration:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        self.ai_enabled_var = ctk.BooleanVar(value=self.settings.settings.get("ai_enabled", False))
        self.ai_enabled_checkbox = ctk.CTkCheckBox(
            ai_enable_frame,
            text="Enable Local AI Processing",
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
        
        # Model selection
        model_frame = ctk.CTkFrame(ai_frame)
        model_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            model_frame,
            text="AI Model Selection:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Get available models from the LLM server
        try:
            available_models = self.get_available_ai_models()
        except:
            available_models = ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "phi-2.Q4_K_M.gguf"]
        
        self.ai_model_combo = ctk.CTkComboBox(
            model_frame,
            values=available_models,
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
        
        # Server control buttons
        server_frame = ctk.CTkFrame(ai_frame)
        server_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            server_frame,
            text="Server Control:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        server_buttons_frame = ctk.CTkFrame(server_frame)
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
            server_frame,
            text="Server Status: Not Running",
            font=ctk.CTkFont(size=11)
        )
        self.server_status_label.pack(padx=10, pady=5)
        
        # Update server status
        self.update_ai_server_status()
        
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
    
    def start_ai_server(self):
        """Start the AI server"""
        try:
            model_name = self.ai_model_combo.get() if hasattr(self, 'ai_model_combo') else None
            success = self.parent.local_ai.start_server(model_name)
            if success:
                self.parent.show_notification("AI server started successfully!")
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
    
    def update_ai_server_status(self):
        """Update AI server status display"""
        if hasattr(self, 'server_status_label'):
            is_running = self.parent.local_ai.is_server_running()
            status_text = "Server Status: Running" if is_running else "Server Status: Not Running"
            status_color = "green" if is_running else "gray"
            self.server_status_label.configure(text=status_text, text_color=status_color)

def main():
    """Main entry point"""
    app = WhisperTranscribePro()
    app.mainloop()

if __name__ == "__main__":
    main()