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
        
    def update(self, audio_data):
        """Update waveform data with new audio"""
        if len(audio_data) > 0:
            # Downsample for visualization
            chunk_size = len(audio_data) // 100
            if chunk_size > 0:
                downsampled = np.array([
                    np.abs(audio_data[i:i+chunk_size]).mean() 
                    for i in range(0, len(audio_data)-chunk_size, chunk_size)
                ])[:100]
                self.waveform_data = downsampled

class HailoIntegration:
    """Optional Hailo AI integration for enhanced features"""
    
    def __init__(self):
        self.hailo_available = self.check_hailo()
        self.face_detection_enabled = False
        
    def check_hailo(self) -> bool:
        """Check if Hailo is available on the system"""
        try:
            result = subprocess.run(
                ['which', 'hailortcli'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
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
            # Run face detection using Hailo
            cmd = f"hailo detect --model yolov8s --input {image_path or 'camera'}"
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse results for face detection
            if "person" in result.stdout.lower():
                return "Speaker detected"
        except:
            pass
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
                    
                    # Update visualizer (throttled)
                    if len(self.audio_data) % 10 == 0:
                        self.visualizer.update(audio_chunk)
                        
                except Exception as e:
                    if "PaErrorCode -9999" in str(e):
                        time.sleep(0.01)
                        continue
                    else:
                        break
            
            stream.stop()
            stream.close()
            
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
                    
                    # Check for Hailo integration
                    if self.hailo.hailo_available and self.settings.settings["hailo_integration"]:
                        speaker = self.hailo.detect_speaker()
                        if speaker:
                            self.ui_queue.put({
                                'type': 'info',
                                'text': f"[Hailo: {speaker}]\n"
                            })
                    
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write(self.text_display.get("1.0", "end-1c"))
            self.show_notification(f"Exported to {filename}")
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
        SettingsWindow(self, self.settings)
    
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
        self.geometry("500x600")
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_settings_ui()
    
    def create_settings_ui(self):
        """Create settings interface"""
        
        # Tabs for different settings categories
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tabview.add("Audio")
        self.tabview.add("Transcription")
        self.tabview.add("Interface")
        self.tabview.add("Advanced")
        
        # Audio Settings Tab
        audio_tab = self.tabview.tab("Audio")
        
        ctk.CTkLabel(audio_tab, text="Audio Settings", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Microphone selection (placeholder)
        ctk.CTkLabel(audio_tab, text="Microphone:").pack(pady=5)
        self.mic_combo = ctk.CTkComboBox(
            audio_tab,
            values=["Default", "USB Microphone"],
            width=300
        )
        self.mic_combo.pack(pady=5)
        self.mic_combo.set("Default")
        
        # Noise reduction
        self.noise_var = ctk.BooleanVar(value=self.settings.settings["noise_reduction"])
        ctk.CTkCheckBox(
            audio_tab,
            text="Enable Noise Reduction",
            variable=self.noise_var
        ).pack(pady=10)
        
        # VAD
        self.vad_var = ctk.BooleanVar(value=self.settings.settings["vad_enabled"])
        ctk.CTkCheckBox(
            audio_tab,
            text="Voice Activity Detection (Auto start/stop)",
            variable=self.vad_var
        ).pack(pady=5)
        
        # Transcription Settings Tab
        trans_tab = self.tabview.tab("Transcription")
        
        ctk.CTkLabel(trans_tab, text="Transcription Settings",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Model selection
        ctk.CTkLabel(trans_tab, text="Whisper Model:").pack(pady=5)
        self.model_combo = ctk.CTkComboBox(
            trans_tab,
            values=["tiny", "base", "small", "medium"],
            width=300
        )
        self.model_combo.pack(pady=5)
        self.model_combo.set(self.settings.settings["model"])
        
        # Language
        ctk.CTkLabel(trans_tab, text="Language:").pack(pady=5)
        self.lang_combo = ctk.CTkComboBox(
            trans_tab,
            values=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            width=300
        )
        self.lang_combo.pack(pady=5)
        self.lang_combo.set(self.settings.settings["language"])
        
        # Hailo Integration
        if self.parent.hailo.hailo_available:
            self.hailo_var = ctk.BooleanVar(value=self.settings.settings["hailo_integration"])
            ctk.CTkCheckBox(
                trans_tab,
                text="Enable Hailo AI Integration (Speaker Detection)",
                variable=self.hailo_var
            ).pack(pady=10)
        
        # Interface Settings Tab
        ui_tab = self.tabview.tab("Interface")
        
        ctk.CTkLabel(ui_tab, text="Interface Settings",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Theme
        ctk.CTkLabel(ui_tab, text="Theme:").pack(pady=5)
        self.theme_combo = ctk.CTkComboBox(
            ui_tab,
            values=["dark", "light", "system"],
            width=300,
            command=self.change_theme
        )
        self.theme_combo.pack(pady=5)
        self.theme_combo.set(self.settings.settings["theme"])
        
        # Window size
        ctk.CTkLabel(ui_tab, text="Window Size:").pack(pady=5)
        self.size_combo = ctk.CTkComboBox(
            ui_tab,
            values=["compact", "standard", "large"],
            width=300
        )
        self.size_combo.pack(pady=5)
        self.size_combo.set(self.settings.settings["window_size"])
        
        # Always on top
        self.top_var = ctk.BooleanVar(value=self.settings.settings["always_on_top"])
        ctk.CTkCheckBox(
            ui_tab,
            text="Always on Top",
            variable=self.top_var
        ).pack(pady=10)
        
        # Font size
        ctk.CTkLabel(ui_tab, text="Font Size:").pack(pady=5)
        self.font_slider = ctk.CTkSlider(
            ui_tab,
            from_=10,
            to=20,
            number_of_steps=10,
            width=300
        )
        self.font_slider.pack(pady=5)
        self.font_slider.set(self.settings.settings["font_size"])
        
        # Save/Cancel buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Save",
            command=self.save_settings
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side="left", padx=5)
    
    def change_theme(self, theme):
        """Live theme preview"""
        if theme in ["dark", "light"]:
            ctk.set_appearance_mode(theme)
    
    def save_settings(self):
        """Save all settings"""
        # Update settings
        self.settings.settings.update({
            "theme": self.theme_combo.get(),
            "model": self.model_combo.get(),
            "language": self.lang_combo.get(),
            "window_size": self.size_combo.get(),
            "always_on_top": self.top_var.get(),
            "font_size": int(self.font_slider.get()),
            "noise_reduction": self.noise_var.get(),
            "vad_enabled": self.vad_var.get(),
        })
        
        if hasattr(self, 'hailo_var'):
            self.settings.settings["hailo_integration"] = self.hailo_var.get()
        
        # Save to file
        self.settings.save_settings()
        
        # Apply changes
        self.parent.apply_theme()
        self.parent.show_notification("Settings saved")
        
        self.destroy()

def main():
    """Main entry point"""
    app = WhisperTranscribePro()
    app.mainloop()

if __name__ == "__main__":
    main()