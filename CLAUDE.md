# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Whisper Transcribe Pi repository.

## Project Overview

Whisper Transcribe Pi is a GUI-based speech-to-text application for Raspberry Pi (and cross-platform) using OpenAI's Whisper model. It provides real-time transcription with a floating window interface, automatic clipboard integration, and one-click installation.

### Key Features
- Real-time speech recognition with GUI
- Floating always-on-top window
- Automatic clipboard integration  
- USB microphone auto-detection
- Desktop menu integration (Pi)
- Cross-platform support (Pi, Mac, Windows, Linux)

## Build/Run Commands

### Raspberry Pi Installation
```bash
# Quick install (recommended)
git clone https://github.com/sethshoultes/whisper-transcribe-pi.git
cd whisper-transcribe-pi
./install.sh

# Run application
./launch_whisper.sh
# OR from menu: Applications → AudioVideo → Whisper Transcribe
```

### Manual Setup (All Platforms)
```bash
# Clone repository
git clone https://github.com/sethshoultes/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Pi-optimized version
python whisper_transcribe.py

# Run universal version (Mac/Windows/Linux)
python whisper_transcribe_universal.py
```

### macOS Specific
```bash
# No additional tools needed - uses built-in pbcopy for clipboard
python whisper_transcribe_universal.py
```

### Windows Specific
```bash
# No additional tools needed - uses built-in clip for clipboard
python whisper_transcribe_universal.py
```

## Test Commands

### Test Audio Input
```bash
# Test microphone detection
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test USB microphone
arecord -l  # Linux/Pi only
arecord -D plughw:2,0 -f S16_LE -r 44100 -d 3 test.wav && aplay test.wav

# Test recording with sounddevice
python3 -c "
import sounddevice as sd
import numpy as np
print('Recording 3 seconds...')
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
sd.wait()
print(f'Recorded {len(audio)} samples')
print(f'Audio level: {np.abs(audio).mean():.4f}')
"
```

### Test Whisper Model
```bash
# Test model loading
python3 -c "import whisper; model = whisper.load_model('tiny'); print('Model loaded successfully')"
```

### Debug Application
```bash
# View debug logs
tail -f /tmp/whisper_debug.log

# Check if app is running
ps aux | grep whisper_transcribe

# Kill existing instances
pkill -f whisper_transcribe.py
```

## Project Structure

```
whisper-transcribe-pi/
├── whisper_transcribe.py          # Main Pi-optimized application
├── whisper_transcribe_universal.py # Cross-platform version
├── launch_whisper.sh              # Launcher script with venv
├── install.sh                     # Automated installer for Pi
├── requirements.txt               # Python dependencies
├── whisper-transcribe.desktop     # Desktop entry for Pi menu
├── icons/                         # Application icons
│   ├── whisper-icon.png         # PNG icon (256x256)
│   ├── whisper-icon.svg         # SVG source
│   └── create_icon.py           # Icon generator script
├── README.md                      # User documentation
├── MARKETING.md                   # Marketing strategy document
├── CLAUDE.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                    # Git ignore rules
```

## Key Technical Decisions

### Audio Processing
- **Sample Rate Handling**: Devices often use 44.1kHz, but Whisper needs 16kHz
  - Solution: Automatic resampling using scipy.signal.resample()
- **USB Mic Detection**: Prioritizes USB mics over built-in audio
  - Searches for 'usb' in device names
  - Falls back to default input if no USB found
- **Buffer Management**: Uses local buffers per recording to avoid race conditions

### Whisper Model Selection
- **Model Choice**: Uses "tiny" model (39MB)
  - Reasons: Pi memory constraints, 2-3s transcription time, good accuracy
  - Alternatives: "base" (74MB), "small" (244MB) for better accuracy but slower

### GUI Architecture
- **Framework**: tkinter (built into Python)
  - Reasons: No extra dependencies, cross-platform, lightweight
- **Window Design**: Floating, always-on-top, semi-transparent
  - 500x400px default size
  - Position: top-left corner (+10+10)
- **Thread Safety**: Background transcription with queue-based UI updates
  - Recording happens in separate thread
  - Transcription in background thread
  - UI updates via queue.Queue() with 50ms polling

### Clipboard Integration
- **Platform Detection**: Uses platform.system() for OS detection
- **Methods**:
  - Linux: xclip with fallback to xsel
  - macOS: pbcopy (built-in)
  - Windows: clip (built-in)
- **Timeout**: 1 second timeout on all clipboard operations

## Common Issues and Solutions

### Audio Issues

**Problem**: "paInvalidSampleRate" error
```python
# Solution: Already handled - automatic sample rate detection
self.device_sample_rate = int(device['default_samplerate'])
# Then resample in transcribe() method
```

**Problem**: "error opening inputS"
```bash
# Check for other instances
pkill -f whisper_transcribe.py
# Verify mic is connected
arecord -l
```

**Problem**: No audio detected
```python
# Check minimum audio threshold
if len(local_audio_data) > 5:  # Minimum chunks for valid audio
```

### Installation Issues

**Problem**: pyaudio build fails
```bash
# Install system dependency
sudo apt install portaudio19-dev
```

**Problem**: scipy not found
```bash
pip install scipy
```

### Performance Issues

**Problem**: Slow transcription
```python
# Use smaller model
model = whisper.load_model("tiny")  # Instead of "base" or larger
```

**Problem**: High memory usage
```python
# Model is loaded once and reused
def load_model(self):
    self.model = whisper.load_model("tiny")  # Cached in memory
```

## Development Guidelines

### Code Style
- Use 4-space indentation
- Follow PEP 8 naming conventions
- Add docstrings for public methods
- Handle exceptions with specific error messages

### Threading Rules
1. Recording happens in separate thread
2. Transcription stays in recording thread (background)
3. UI updates only via queue
4. Never update tkinter from background threads directly

### Error Handling Pattern
```python
try:
    # Risky operation
    stream = sd.InputStream(...)
except Exception as e:
    print(f"Error: {str(e)[:50]}")  # Truncate long errors
    self.status.config(text="Error occurred")
    # Graceful degradation
```

### Platform-Specific Code
```python
# Always check platform before using OS-specific features
if platform.system() == "Darwin":  # macOS
    subprocess.run(['pbcopy'], ...)
elif platform.system() == "Windows":
    subprocess.run(['clip'], ...)
else:  # Linux/Unix
    subprocess.run(['xclip'], ...)
```

## Performance Optimization

### Audio Processing
- **Blocksize**: 512 samples (balanced latency vs CPU)
- **Skip ALSA errors**: Continue on -9999 errors
- **Local buffers**: Avoid shared state between recordings

### Whisper Optimization
- **Language hint**: language="en" speeds up processing
- **FP16 disabled**: fp16=False for CPU compatibility
- **Model caching**: Load once, reuse for all transcriptions

### UI Responsiveness
- **Queue polling**: 50ms interval for smooth updates
- **Background threads**: Never block main UI thread
- **Status updates**: Immediate feedback for all actions

## Platform-Specific Considerations

### Raspberry Pi
- Optimized for Pi 4/5 hardware
- Handles USB mic quirks
- Desktop integration via .desktop file
- Assumes Debian-based OS

### macOS
- Window transparency may behave differently
- Command key bindings could be added
- Retina display scaling handled by tkinter

### Windows
- Paths use forward slashes (Python handles conversion)
- No sudo required for installation
- Windows Defender may flag first run

### Linux (Generic)
- Requires X11 or Wayland with XWayland
- May need to install xclip: `sudo apt install xclip`
- Different audio systems (ALSA, PulseAudio, PipeWire) handled by sounddevice

## Important Files

### Core Application
- `whisper_transcribe.py`: Main app logic, Pi-optimized
- `whisper_transcribe_universal.py`: Cross-platform version with UniversalClipboard class

### Configuration
- `requirements.txt`: Python package versions
- `.gitignore`: Excludes venv, cache, audio files

### Installation
- `install.sh`: Automated Pi installer
- `launch_whisper.sh`: Venv activation and launch
- `whisper-transcribe.desktop`: Menu integration

## Integration Examples

### Voice-Controlled Pi Projects
```python
# Use transcribed text for commands
if "lights on" in transcription.lower():
    GPIO.output(LED_PIN, GPIO.HIGH)
```

### Accessibility Tools
```python
# Auto-paste to active window
UniversalClipboard.copy(text)
# User presses Ctrl+V to paste
```

### Education/Research
```python
# Save all transcriptions with timestamps
with open("lecture_notes.txt", "a") as f:
    f.write(f"[{timestamp}] {text}\n")
```

## Debugging Tips

1. **Check logs first**: `/tmp/whisper_debug.log`
2. **Verify audio device**: `sd.query_devices()`
3. **Test model separately**: Load model in Python REPL
4. **Monitor memory**: `htop` or `top` during transcription
5. **Check clipboard**: Test clipboard commands manually

## Future Improvements to Consider

- [ ] Add language selection dropdown
- [ ] Support for larger Whisper models (user choice)
- [ ] Hotkey customization
- [ ] Dark mode theme
- [ ] Export transcriptions to various formats
- [ ] Real-time streaming transcription
- [ ] Multi-microphone support
- [ ] Speaker diarization

## Testing Checklist

When making changes, test:
- [ ] Installation on fresh Pi
- [ ] USB mic detection
- [ ] Recording starts/stops
- [ ] Transcription appears in window
- [ ] Clipboard copy works
- [ ] Desktop menu entry works
- [ ] Cross-platform compatibility
- [ ] Memory usage stays stable
- [ ] No zombie processes after exit

## Support Resources

- GitHub Issues: https://github.com/sethshoultes/whisper-transcribe-pi/issues
- Whisper Documentation: https://github.com/openai/whisper
- Raspberry Pi Forums: https://forums.raspberrypi.com/
- OpenAI Community: https://community.openai.com/