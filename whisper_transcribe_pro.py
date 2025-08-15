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
logging.basicConfig(
    level=logging.INFO,
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
    """Optional Hailo AI integration for enhanced features"""
    
    def __init__(self):
        self.hailo_available = self.check_hailo()
        # Enable face detection if Hailo is available
        self.face_detection_enabled = self.hailo_available
        # Path to the real Hailo detection system
        self.hailo_script_path = os.path.expanduser("~/hailo-ai/scripts/simple_photo_detect.sh")
        
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
    
    def detect_speaker(self, image_path: Optional[str] = None) -> Optional[str]:
        """Use Hailo for speaker detection via face detection"""
        if not self.hailo_available or not self.face_detection_enabled:
            return None
            
        try:
            # Use the real Hailo detection system
            # The simple_photo_detect.sh script captures and detects
            cmd = ["bash", self.hailo_script_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,  # Give more time for actual detection
                cwd=os.path.expanduser("~/hailo-ai")
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                # Parse real Hailo output for person/face detection
                # The script outputs detection results
                if "person" in output.lower() or "people" in output.lower():
                    # Extract number of people detected if available
                    import re
                    match = re.search(r'(\d+)\s*(person|people)', output.lower())
                    if match:
                        count = int(match.group(1))
                        return f"{count} speaker{'s' if count > 1 else ''} detected"
                    # Fallback if we detect person but can't parse count
                    if "person" in output.lower():
                        return "Speaker detected"
        except subprocess.TimeoutExpired:
            logging.warning("Hailo detection timed out")
        except Exception as e:
            logging.debug(f"Hailo detection error: {e}")
        return None

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
            "window_opacity": 0.95,
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
            }
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
        
        # Window transparency
        try:
            self.attributes("-alpha", self.settings.settings["window_opacity"])
        except:
            pass
        
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
            text="🎙️ Whisper Transcribe Pro",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Status Indicator
        self.status_label = ctk.CTkLabel(
            self.top_frame,
            text="● Ready",
            font=ctk.CTkFont(size=12),
            text_color="green"
        )
        self.status_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        
        # Settings Button
        self.settings_btn = ctk.CTkButton(
            self.top_frame,
            text="⚙️",
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
            text="🎤 Click to Record",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            corner_radius=30,
            fg_color="#4299E1",
            hover_color="#3182CE",
            command=self.toggle_recording
        )
        self.record_button.grid(row=1, column=0, padx=20, pady=15, sticky="ew")
        
        # Transcription Display
        self.text_frame = ctk.CTkFrame(self.main_frame)
        self.text_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        self.text_frame.grid_columnconfigure(0, weight=1)
        self.text_frame.grid_rowconfigure(0, weight=1)
        
        self.text_display = ctk.CTkTextbox(
            self.text_frame,
            font=ctk.CTkFont(size=self.settings.settings["font_size"]),
            wrap="word"
        )
        self.text_display.grid(row=0, column=0, sticky="nsew")
        
        # Action Buttons Frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure((0,1,2,3,4), weight=1)
        
        # Action Buttons with icons
        buttons = [
            ("📋 Copy Last", self.copy_last),
            ("📋 Copy All", self.copy_all),
            ("💾 Export", self.export_transcription),
            ("🔍 Search", self.search_transcription),
            ("🗑️ Clear", self.clear_transcriptions)
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
        
        if self.hailo.hailo_available:
            self.hailo_label = ctk.CTkLabel(
                self.bottom_frame,
                text="Hailo: ✓",
                font=ctk.CTkFont(size=10),
                text_color="green"
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
            self.update_status("● Ready", "green")
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
            text="⏹️ Stop Recording",
            fg_color="#EF4444",
            hover_color="#DC2626"
        )
        self.update_status("● Recording...", "red")
        
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
            text="🎤 Click to Record",
            fg_color="#4299E1",
            hover_color="#3182CE"
        )
        self.update_status("Processing...", "orange")
        
        # Hide visualizer
        self.visualizer_frame.grid_remove()
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
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
                
                # Transcribe
                result = self.model.transcribe(
                    tmp.name,
                    language=self.settings.settings["language"],
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
                    
                    self.update_status("● Ready", "green")
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
        self.transcription_history.clear()
        self.update_status("Cleared", "green")
    
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
        self.after(3000, lambda: self.update_status("● Ready", "green"))
    
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
                    
        except queue.Empty:
            pass
        
        self.after(50, self.process_ui_queue)

class SettingsWindow(ctk.CTkToplevel):
    """Settings window with comprehensive options"""
    
    def __init__(self, parent, settings):
        super().__init__(parent)
        
        self.parent = parent
        self.settings = settings
        
        self.title("Settings")
        self.geometry("600x650")
        
        # Center the window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.winfo_screenheight() // 2) - (650 // 2)
        self.geometry(f"600x650+{x}+{y}")
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Get available audio devices
        self.audio_devices = self.get_audio_devices()
        
        self.create_settings_ui()
    
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
        
        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="⚙️ Settings",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(10, 5))
        
        # Tabs for different settings categories
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Create tabs
        self.tabview.add("🎤 Audio")
        self.tabview.add("💬 Transcription")
        self.tabview.add("🎨 Interface")
        self.tabview.add("⚡ Advanced")
        
        # Audio Settings Tab
        audio_tab = self.tabview.tab("🎤 Audio")
        audio_frame = ctk.CTkScrollableFrame(audio_tab, height=400)
        audio_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
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
        ctk.CTkButton(
            mic_frame,
            text="🎙️ Test Microphone",
            command=self.test_microphone,
            width=200,
            height=35
        ).pack(pady=10)
        
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
        trans_tab = self.tabview.tab("💬 Transcription")
        trans_frame = ctk.CTkScrollableFrame(trans_tab, height=400)
        trans_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
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
                text="Enable Hailo AI for Speaker Detection",
                variable=self.hailo_var,
                font=ctk.CTkFont(size=12)
            )
            hailo_check.pack(anchor="w", padx=20, pady=5)
            
            ctk.CTkLabel(
                hailo_frame,
                text="✅ Hailo hardware detected and ready",
                font=ctk.CTkFont(size=10),
                text_color="green"
            ).pack(anchor="w", padx=20)
        
        # Interface Settings Tab
        ui_tab = self.tabview.tab("🎨 Interface")
        ui_frame = ctk.CTkScrollableFrame(ui_tab, height=400)
        ui_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
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
        
        # Window opacity
        ctk.CTkLabel(
            window_frame,
            text=f"Window Opacity: {int(self.settings.settings.get('window_opacity', 0.95) * 100)}%",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=20, pady=5)
        
        self.opacity_slider = ctk.CTkSlider(
            window_frame,
            from_=0.5,
            to=1.0,
            number_of_steps=10,
            width=300,
            command=self.update_opacity_label
        )
        self.opacity_slider.pack(anchor="w", padx=30, pady=5)
        self.opacity_slider.set(self.settings.settings.get("window_opacity", 0.95))
        
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
        
        # Advanced Settings Tab
        adv_tab = self.tabview.tab("⚡ Advanced")
        adv_frame = ctk.CTkScrollableFrame(adv_tab, height=400)
        adv_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
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
        
        self.debug_var = ctk.BooleanVar(value=False)
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
        
        # Save/Cancel buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=15, pady=15)
        button_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkButton(
            button_frame,
            text="💾 Save Settings",
            command=self.save_settings,
            width=150,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        ).grid(row=0, column=0, padx=5, sticky="e")
        
        ctk.CTkButton(
            button_frame,
            text="🔄 Reset Defaults",
            command=self.reset_defaults,
            width=150,
            height=40,
            font=ctk.CTkFont(size=13)
        ).grid(row=0, column=1, padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="❌ Cancel",
            command=self.destroy,
            width=150,
            height=40,
            font=ctk.CTkFont(size=13)
        ).grid(row=0, column=2, padx=5, sticky="w")
    
    def test_microphone(self):
        """Test microphone with audio recording"""
        try:
            import sounddevice as sd
            import numpy as np
            
            # Get device index from selection
            device_str = self.mic_combo.get()
            device_index = int(device_str.split(":")[0]) if ":" in device_str else None
            
            # Record 2 seconds
            duration = 2
            sample_rate = 44100
            
            self.parent.show_notification("Recording 2 second test...")
            audio = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1,
                          device=device_index,
                          dtype=np.float32)
            sd.wait()
            
            # Check audio level
            level = np.abs(audio).mean()
            if level > 0.001:
                self.parent.show_notification(f"✅ Microphone working! Level: {level:.4f}")
            else:
                self.parent.show_notification("⚠️ No audio detected. Check microphone.")
        except Exception as e:
            self.parent.show_notification(f"❌ Error: {str(e)[:50]}")
    
    def preview_theme(self, theme_name):
        """Live theme preview"""
        theme_map = {"Dark Mode": "dark", "Light Mode": "light", "System Default": "system"}
        theme = theme_map.get(theme_name, "dark")
        if theme != "system":
            ctk.set_appearance_mode(theme)
    
    def update_opacity_label(self, value):
        """Update opacity label with slider value"""
        # Find and update the opacity label
        pass  # Label updates are handled differently in CTk
    
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
        self.opacity_slider.set(0.95)
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
            "window_opacity": self.opacity_slider.get(),
            "noise_reduction": self.noise_var.get(),
            "vad_enabled": self.vad_var.get(),
            "audio_device": device_text,
            "sample_rate": sample_rate,
            "debug_logging": self.debug_var.get() if hasattr(self, 'debug_var') else False
        })
        
        if hasattr(self, 'hailo_var'):
            self.settings.settings["hailo_integration"] = self.hailo_var.get()
        
        # Save to file
        self.settings.save_settings()
        
        # Apply changes
        self.parent.apply_theme()
        
        # Apply font size to text display
        new_font_size = int(self.font_slider.get())
        self.parent.text_display.configure(font=ctk.CTkFont(size=new_font_size))
        
        # Apply window changes
        if self.top_var.get():
            self.parent.attributes("-topmost", True)
        else:
            self.parent.attributes("-topmost", False)
        
        # Apply opacity
        try:
            self.parent.attributes("-alpha", self.opacity_slider.get())
        except:
            pass
        
        self.parent.show_notification("✅ Settings saved successfully!")
        
        # Note about model change
        if model != self.parent.settings.settings.get("model", "tiny"):
            self.parent.show_notification("⚠️ Restart app to load new model")
        
        self.destroy()

def main():
    """Main entry point"""
    app = WhisperTranscribePro()
    app.mainloop()

if __name__ == "__main__":
    main()