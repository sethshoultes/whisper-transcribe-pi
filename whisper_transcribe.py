#!/usr/bin/env python3

import os
import sys
import time
import threading
import queue
import logging
import numpy as np
import sounddevice as sd
import whisper
import tempfile
import wave
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import ttk

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('/tmp/whisper_debug.log'),
        logging.StreamHandler()
    ]
)

class WorkingWhisper:
    def __init__(self):
        self.model = None
        self.recording = False
        self.audio_data = []
        self.whisper_sample_rate = 16000  # Whisper needs 16kHz
        self.device_sample_rate = None  # Will be set based on device
        self.last_transcribed_text = None  # Track last text to avoid duplicates
        self.transcription_count = 0  # Track transcription number
        self.ui_queue = queue.Queue()  # Queue for thread-safe UI updates
        
        # Find microphone with better device detection
        devices = sd.query_devices()
        self.device_index = None
        
        # First, try to find USB microphone
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if 'usb' in device['name'].lower():
                    self.device_index = i
                    self.device_sample_rate = int(device['default_samplerate'])
                    print(f"Using USB mic: {device['name']} @ {self.device_sample_rate}Hz")
                    break
        
        # Fallback to default input device if no USB found
        if self.device_index is None:
            try:
                self.device_index = sd.default.device[0]  # Default input
                default_device = devices[self.device_index]
                self.device_sample_rate = int(default_device['default_samplerate'])
                print(f"Using default mic: {default_device['name']} @ {self.device_sample_rate}Hz")
            except:
                # Last resort - find any input device
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        self.device_index = i
                        self.device_sample_rate = int(device['default_samplerate'])
                        print(f"Using first available mic: {device['name']} @ {self.device_sample_rate}Hz")
                        break
        
        # Default to 44100 if still not set
        if self.device_sample_rate is None:
            self.device_sample_rate = 44100
        
        self.setup_ui()
        self.load_model()
    
    def safe_type_text(self, text):
        """Deprecated - we now display in window instead of auto-typing"""
        # No longer used - keeping for compatibility
        return False
    
    def setup_ui(self):
        """Create floating window with transcription display"""
        self.root = tk.Tk()
        self.root.title("Whisper Transcription")
        self.root.geometry("500x400")
        self.root.attributes('-topmost', True)  # Always on top
        
        # Move to corner
        self.root.geometry("+10+10")
        
        # Status label
        self.status = tk.Label(self.root, text="Loading...", font=("Arial", 10))
        self.status.pack(pady=5)
        
        # Record button
        self.btn = tk.Button(
            self.root,
            text="🎤 Click or Press [R] to Record",
            command=self.toggle_recording,
            font=("Arial", 12),
            bg="green",
            fg="white",
            height=2
        )
        self.btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Transcription display area
        tk.Label(self.root, text="Transcriptions:", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        
        # Create scrollable text area with fixed height
        from tkinter import scrolledtext
        text_frame = tk.Frame(self.root, height=200)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        text_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        self.text_display = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Button frame for copy/clear
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Copy last button
        self.copy_btn = tk.Button(
            button_frame,
            text="📋 Copy Last",
            command=self.copy_last_transcription,
            font=("Arial", 9)
        )
        self.copy_btn.pack(side=tk.LEFT, padx=2)
        
        # Copy all button
        self.copy_all_btn = tk.Button(
            button_frame,
            text="📋 Copy All",
            command=self.copy_all_transcriptions,
            font=("Arial", 9)
        )
        self.copy_all_btn.pack(side=tk.LEFT, padx=2)
        
        # Clear button
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_transcriptions,
            font=("Arial", 9)
        )
        self.clear_btn.pack(side=tk.LEFT, padx=2)
        
        # Bind keyboard shortcuts
        self.root.bind('<r>', lambda e: self.toggle_recording())
        self.root.bind('<R>', lambda e: self.toggle_recording())
        self.root.bind('<space>', lambda e: self.toggle_recording())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        # Make window semi-transparent
        self.root.attributes('-alpha', 0.95)
    
    def copy_last_transcription(self):
        """Copy the last transcription to clipboard"""
        # Run in thread to avoid freezing
        def do_copy():
            if self.last_transcribed_text:
                try:
                    subprocess.run(
                        ['xclip', '-selection', 'clipboard'],
                        input=self.last_transcribed_text.encode(),
                        check=True,
                        capture_output=True,
                        timeout=1
                    )
                    self.root.after(0, lambda: self.status.config(text="📋 Copied last transcription"))
                except:
                    self.root.after(0, lambda: self.status.config(text="⚠️ Could not copy"))
        
        threading.Thread(target=do_copy, daemon=True).start()
    
    def copy_all_transcriptions(self):
        """Copy all transcriptions to clipboard"""
        def do_copy():
            text = self.text_display.get("1.0", tk.END).strip()
            if text:
                try:
                    subprocess.run(
                        ['xclip', '-selection', 'clipboard'],
                        input=text.encode(),
                        check=True,
                        capture_output=True,
                        timeout=1
                    )
                    self.root.after(0, lambda: self.status.config(text="📋 Copied all transcriptions"))
                except:
                    self.root.after(0, lambda: self.status.config(text="⚠️ Could not copy"))
        
        threading.Thread(target=do_copy, daemon=True).start()
    
    def clear_transcriptions(self):
        """Clear the transcription display"""
        self.text_display.delete("1.0", tk.END)
        self.status.config(text="Cleared transcriptions")
    
    def load_model(self):
        """Load Whisper model in background"""
        def load():
            self.model = whisper.load_model("tiny")
            self.status.config(text="Ready! Click or press R")
            self.btn.config(state=tk.NORMAL)
        
        self.btn.config(state=tk.DISABLED)
        threading.Thread(target=load, daemon=True).start()
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.model:
            return
        
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording"""
        logging.info(f"START RECORDING - count is {self.transcription_count}")
        self.recording = True
        self.btn.config(text="⏹️ Recording... Click to Stop", bg="red")
        self.status.config(text="🎤 Speak now...")
        
        def record():
            # Use LOCAL buffer to avoid race conditions!
            local_audio_data = []
            
            try:
                stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=self.device_sample_rate,  # Use device's native sample rate
                    channels=1,
                    dtype=np.float32,
                    blocksize=512
                )
                
                stream.start()
                while self.recording:
                    try:
                        audio_chunk = stream.read(512)[0]
                        local_audio_data.append(audio_chunk)  # Use LOCAL buffer!
                    except Exception as e:
                        print(f"Audio read error: {e}")
                        # Don't break on ALSA errors, just skip this chunk
                        if "PaErrorCode -9999" in str(e):
                            time.sleep(0.01)  # Brief pause before retry
                            continue
                        else:
                            break
                
                stream.stop()
                stream.close()
                
                if len(local_audio_data) > 5:  # Minimum chunks for valid audio
                    audio_array = np.concatenate(local_audio_data, axis=0).flatten()
                    # Keep transcription in background thread to avoid freezing
                    self.transcribe(audio_array)
                else:
                    self.status.config(text="Recording too short")
                    
            except Exception as e:
                self.status.config(text=f"Recording error: {str(e)[:20]}")
                self.recording = False
                self.btn.config(text="🎤 Click or Press [R] to Record", bg="green")
        
        threading.Thread(target=record, daemon=True).start()
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        self.btn.config(text="🎤 Click or Press [R] to Record", bg="green")
        self.status.config(text="Transcribing...")
    
    def transcribe(self, audio_data):
        """Transcribe audio"""
        import traceback
        from scipy import signal
        logging.info(f"TRANSCRIBE CALLED from: {traceback.extract_stack()[-2]}")
        logging.info(f"Audio data length: {len(audio_data)}")
        
        # Resample if needed
        if self.device_sample_rate != self.whisper_sample_rate:
            # Calculate the number of samples for the target rate
            num_samples = int(len(audio_data) * self.whisper_sample_rate / self.device_sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            logging.info(f"Resampled from {self.device_sample_rate}Hz to {self.whisper_sample_rate}Hz")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.whisper_sample_rate)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            try:
                result = self.model.transcribe(
                    tmp.name,
                    language="en",
                    fp16=False
                )
                text = result["text"].strip()
                
                if text:
                    # Check if this is a duplicate of the last transcription FIRST
                    if text == self.last_transcribed_text:
                        print(f"⚠️ Duplicate text detected, skipping: {text[:30]}...")
                        self.status.config(text="Duplicate detected, skipping")
                        return
                    
                    # NOW increment counter and store text
                    self.transcription_count += 1
                    self.last_transcribed_text = text
                    
                    print(f"\n🎯 NEW TRANSCRIPTION #{self.transcription_count}: {repr(text[:50])}...")
                    print(f"   ID of text object: {id(text)}")
                    print(f"   Stored as last_text: {repr(self.last_transcribed_text[:30]) if self.last_transcribed_text else 'None'}...")
                    
                    logging.info(f"About to copy to clipboard for #{self.transcription_count}")
                    
                    # Copy to clipboard - skip wl-copy as it hangs on this system
                    clipboard_copied = False
                    try:
                        # Use xclip directly (wl-copy hangs on this Pi)
                        subprocess.run(
                            ['xclip', '-selection', 'clipboard'],
                            input=text.encode(),
                            check=True,
                            capture_output=True,
                            timeout=0.5  # Short timeout
                        )
                        clipboard_copied = True
                        print("📋 Copied to clipboard")
                    except Exception as e:
                        print(f"⚠️ Could not copy to clipboard: {e}")
                    
                    # Queue the transcription for UI update (thread-safe)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    display_text = f"[{timestamp}] {text}\n\n"
                    
                    # Create a local copy to avoid closure issues
                    current_text = str(text)
                    current_count = self.transcription_count
                    
                    print(f"🔵 QUEUEING transcription #{current_count}: {current_text[:30]}...")
                    
                    # Put update data in queue for main thread to process IMMEDIATELY
                    self.ui_queue.put({
                        'type': 'transcription',
                        'text': display_text,
                        'status': f"✅ Transcribed #{current_count}" + (" (in clipboard)" if clipboard_copied else ""),
                        'transcription_count': current_count,
                        'actual_text': current_text  # Add for debugging
                    })
                    
                    print(f"📨 QUEUED transcription #{self.transcription_count}")
                    
                    # Save to file
                    with open(os.path.expanduser("~/whisper_history.txt"), "a") as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
                else:
                    self.status.config(text="No speech detected")
                    
            except Exception as e:
                self.status.config(text=f"Error: {str(e)[:30]}")
            finally:
                os.unlink(tmp.name)
    
    def process_ui_queue(self):
        """Process queued UI updates from background threads"""
        try:
            while True:
                # Get all pending updates without blocking
                update = self.ui_queue.get_nowait()
                
                if update['type'] == 'transcription':
                    print(f"📥 PROCESSING transcription #{update['transcription_count']}: {update.get('actual_text', '')[:30]}...")
                    
                    # Update the text display
                    self.text_display.insert(tk.END, update['text'])
                    self.text_display.see(tk.END)
                    
                    # Update status
                    self.status.config(text=update['status'])
                    
                    print(f"✅ UI DISPLAYED transcription #{update['transcription_count']}")
        except queue.Empty:
            pass
        
        # Schedule next check - reduced to 50ms for faster updates
        self.root.after(50, self.process_ui_queue)
    
    def run(self):
        """Run the application"""
        print("🎙️ Whisper Running!")
        print("• Click button or press R to record")
        print("• Transcriptions appear in window")
        print("• Also copied to clipboard")
        print("• Press ESC to exit\n")
        
        # Clear any stale queue items
        while not self.ui_queue.empty():
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start processing UI updates immediately
        self.root.after(10, self.process_ui_queue)
        
        self.root.mainloop()

def main():
    # Install xdotool if needed
    try:
        subprocess.run(['which', 'xdotool'], check=True, capture_output=True)
    except:
        print("Installing xdotool for keyboard simulation...")
        subprocess.run(['sudo', 'apt', 'install', '-y', 'xdotool'])
    
    app = WorkingWhisper()
    app.run()

if __name__ == "__main__":
    main()