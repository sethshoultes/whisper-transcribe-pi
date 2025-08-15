# Whisper Transcribe Pi

A lightweight speech-to-text application for Raspberry Pi using OpenAI's Whisper model. Features a simple GUI interface and clipboard integration for easy transcription.

## Whisper Transcribe Screenshots
<img width="598" height="524" alt="whisper-1" src="https://github.com/user-attachments/assets/e00e2f21-e5f1-45ee-8c88-d99ab1b5cc0b" />

<img width="603" height="524" alt="whisper-2" src="https://github.com/user-attachments/assets/f0991feb-367e-487b-92af-519f3d88063e" />

## Features

- **Real-time Speech Recognition** - Click to record, automatic transcription
- **Clipboard Integration** - Transcribed text automatically copied to clipboard
- **Floating Window** - Always-on-top window for easy access
- **Keyboard Shortcuts** - Press 'R' or spacebar to start recording
- **Transcription History** - View and copy previous transcriptions
- **USB Microphone Support** - Automatic detection and configuration

## Requirements

### Raspberry Pi
- Raspberry Pi 4 or 5 (recommended: 4GB+ RAM)
- Raspberry Pi OS with Desktop
- USB Microphone
- Python 3.7+
- Internet connection (for initial setup)

### Other Platforms (Mac/Windows/Linux)
- Any modern computer with microphone
- Python 3.7+
- macOS 10.14+, Windows 10+, or Linux with GUI
- Use `whisper_transcribe_universal.py` for cross-platform support

## Quick Installation

### Option 1: Automated Install

```bash
# Clone the repository
git clone https://github.com/yourusername/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Run the installer
chmod +x install.sh
./install.sh
```

### Option 2: Manual Install

```bash
# Install system dependencies
sudo apt update
sudo apt install -y portaudio19-dev python3-pip python3-venv

# Clone the repository
git clone https://github.com/yourusername/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Make launcher executable
chmod +x launch_whisper.sh
```

## Usage

### Running the Application

**From Terminal:**
```bash
./launch_whisper.sh
```

**From Desktop:**
After installation, find "Whisper Transcribe" in your Applications menu under AudioVideo.

### How to Use

1. **Start Recording**: Click the green button or press 'R'
2. **Speak**: The button turns red while recording
3. **Stop Recording**: Click the button again
4. **View Transcription**: Text appears in the window
5. **Copy Text**: Use "Copy Last" or "Copy All" buttons

### Keyboard Shortcuts

- `R` or `Space` - Start/stop recording
- `Escape` - Exit application

## Configuration

The application automatically detects your USB microphone. If you have multiple audio devices, it will prefer USB microphones over built-in audio.

### Sample Rate Configuration

The app automatically handles different microphone sample rates. Most USB microphones use 44.1kHz, which is automatically resampled to 16kHz for Whisper.

## Troubleshooting

### "Error opening inputS" or Sample Rate Errors

This is automatically handled by the application. If you still encounter issues:

```bash
# Check your audio devices
arecord -l

# Test your microphone
arecord -D plughw:2,0 -f S16_LE -r 44100 -d 3 test.wav
aplay test.wav
```

### Application Won't Start

```bash
# Check if another instance is running
pkill -f whisper_transcribe.py

# Reinstall dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### No Audio Detected

1. Check USB microphone is connected
2. Verify it's detected: `lsusb | grep -i audio`
3. Test with system tools: `arecord -l`

## Performance

- **Raspberry Pi 5**: ~2-3 seconds per transcription
- **Raspberry Pi 4**: ~4-6 seconds per transcription
- **Model**: Uses Whisper "tiny" model for optimal Pi performance

## Project Structure

```
whisper-transcribe-pi/
├── whisper_transcribe.py    # Main application
├── launch_whisper.sh         # Launcher script
├── install.sh               # Installation script
├── requirements.txt         # Python dependencies
├── whisper-transcribe.desktop # Desktop entry
├── icons/                   # Application icons
│   ├── whisper-icon.png
│   └── whisper-icon.svg
└── README.md               # This file
```

## Building from Source

```bash
# Clone repository
git clone https://github.com/yourusername/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python3 whisper_transcribe.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [SoundDevice](https://python-sounddevice.readthedocs.io/) for audio recording
- Built specifically for Raspberry Pi enthusiasts

## Author

Created for the Raspberry Pi community to enable easy speech-to-text functionality on Pi devices.

---

**Note**: This application is optimized for Raspberry Pi hardware and may need adjustments for other platforms.
