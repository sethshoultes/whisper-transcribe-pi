# Whisper Transcribe Pi

A professional speech-to-text application for Raspberry Pi using OpenAI's Whisper model. Available in both Standard and Pro versions with advanced features, modern UI, and comprehensive AI integration supporting Local AI, Claude API, and OpenAI API.

## Whisper Transcribe Screenshots
<img width="598" height="524" alt="whisper-1" src="https://github.com/user-attachments/assets/e00e2f21-e5f1-45ee-8c88-d99ab1b5cc0b" />

<img width="603" height="524" alt="whisper-2" src="https://github.com/user-attachments/assets/f0991feb-367e-487b-92af-519f3d88063e" />

## Version Comparison

### Standard Version (`whisper_transcribe.py`)
- Simple, lightweight GUI with tkinter
- Real-time speech recognition
- Clipboard integration
- USB microphone support
- Basic transcription history
- Keyboard shortcuts (R/Space to record)

### Pro Version (`whisper_transcribe_pro.py`) - NEW!
- **Modern UI** with CustomTkinter framework
- **Dark/Light themes** with live preview
- **Comprehensive settings** with 4-tab interface
- **Inline microphone testing** with waveform visualization
- **AI Integration** - Local AI, Claude API, OpenAI API support
- **Auto-send & Manual AI processing** modes
- **AI-powered audio enhancement** (Hailo integration)
- **Noise reduction & Voice Activity Detection**
- **Export functionality** with timestamped files
- **Multiple Whisper models** (tiny to medium)
- **Multi-language support** (12 languages)
- **Real-time audio visualization**
- **Customizable window sizes and fonts**
- **Professional status indicators**
- **Advanced audio processing**

## Requirements

### Raspberry Pi
- Raspberry Pi 4 or 5 (recommended: 4GB+ RAM)
- Raspberry Pi OS with Desktop
- USB Microphone
- Python 3.7+
- Internet connection (for initial setup)

### AI Integration Requirements

#### For Local AI (Optional)
- **Additional RAM**: 8GB recommended for larger models
- **Storage Space**: 0.6-4GB per model
- **Dependencies**: Auto-installed during setup
- **Models**: TinyLlama (638MB), Phi-2 (1.4GB), Mistral-7B (4GB)

#### For Claude API (Optional)
- **API Key**: Free account at console.anthropic.com
- **Dependencies**: `pip install anthropic`
- **Internet**: Required for API calls
- **Cost**: Pay-per-use (starting ~$0.25/1M tokens)

#### For OpenAI API (Optional)
- **API Key**: Account at platform.openai.com
- **Dependencies**: `pip install openai`
- **Internet**: Required for API calls
- **Cost**: Pay-per-use (GPT-3.5: ~$0.50/1M tokens, GPT-4: ~$10/1M tokens)

### Optional: Hailo AI Accelerator
- Hailo-8 AI processor (for enhanced audio processing)
- Hailo software stack installed


## Quick Installation

### Option 1: Pro Version (Recommended)

```bash
# Clone the repository
git clone https://github.com/sethshoultes/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Run the Pro installer
chmod +x setup_pro.sh
./setup_pro.sh
```

### Option 2: Standard Version

```bash
# Clone the repository
git clone https://github.com/sethshoultes/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Run the standard installer
chmod +x install.sh
./install.sh
```

### Option 3: Manual Install

```bash
# Install system dependencies
sudo apt update
sudo apt install -y portaudio19-dev python3-pip python3-venv

# Clone the repository
git clone https://github.com/sethshoultes/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# For Pro version, make launcher executable
chmod +x launch_whisper_pro.sh

# For Standard version
chmod +x launch_whisper.sh
```

## Usage

### Running the Application

**Pro Version:**
```bash
./launch_whisper_pro.sh
```
Or find "Whisper Transcribe Pro" in your Applications menu.

**Standard Version:**
```bash
./launch_whisper.sh
```
Or find "Whisper Transcribe" in your Applications menu under AudioVideo.

### How to Use

1. **Start Recording**: Click the record button or press 'R'/'Space'
2. **Speak**: The button/status changes while recording
3. **Stop Recording**: Click again or release key
4. **View Transcription**: Text appears in the window
5. **Copy/Export**: Use the action buttons below the text

### Pro Version Settings

Access comprehensive settings via the Settings button:

#### Audio Tab
- Select microphone from available devices
- Test microphone with inline waveform display
- Enable noise reduction
- Configure Voice Activity Detection
- Set sample rate (16kHz, 44.1kHz, 48kHz)

#### Transcription Tab
- Choose Whisper model (tiny/base/small/medium)
- Select language (12 languages supported)
- Enable Hailo AI enhancement
- Configure auto-transcription settings

#### Interface Tab
- Switch between Dark/Light themes
- Adjust window size (Compact/Standard/Large)
- Configure font size (10-20pt)
- Toggle always-on-top
- Enable compact mode

#### AI Integration Tab
- Choose AI provider (Local AI, Claude API, OpenAI API)
- Configure API keys for cloud services
- Download and manage local AI models
- Enable auto-send or manual AI processing
- Start/stop local AI server

#### Advanced Tab
- Configure hotkeys
- Enable debug logging
- View system information
- Access log files

### Keyboard Shortcuts

- `R` or `Space` - Start/stop recording
- `Escape` - Exit application
- Customizable in Pro version settings

## AI Integration Features

### AI Provider Support
The Pro version includes comprehensive AI integration with support for three different providers:

#### 1. Local AI (Private & Offline)
- **TinyLlama 1.1B** - Lightweight model optimized for Raspberry Pi
- **Phi-2** - Microsoft's efficient 2.7B parameter model  
- **Mistral-7B** - High-quality 7B parameter model
- **Privacy**: All processing happens locally on your device
- **Cost**: Free after initial download
- **Requirements**: Additional storage space (0.6-4GB per model)

#### 2. Claude API (Anthropic)
- **Claude Haiku** - Fast and cost-effective model
- **Privacy**: Data sent to Anthropic's servers
- **Cost**: Pay-per-use API pricing
- **Requirements**: Anthropic API key from console.anthropic.com

#### 3. OpenAI API
- **GPT-3.5 Turbo** - Fast and affordable
- **GPT-4** - Most capable model
- **GPT-4 Turbo** - Enhanced version of GPT-4
- **GPT-4o & GPT-4o Mini** - Latest optimized models
- **Privacy**: Data sent to OpenAI's servers
- **Cost**: Pay-per-use API pricing
- **Requirements**: OpenAI API key from platform.openai.com

### AI Configuration

#### Setting Up Local AI
1. Open Settings → AI Integration tab
2. Select "Local AI (Private & Offline)" as provider
3. Choose your preferred model from the dropdown
4. Click "Download TinyLlama" if not already available (638 MB)
5. Click "Start AI Server" to begin local processing

#### Setting Up Claude API
1. Get your API key from https://console.anthropic.com
2. Open Settings → AI Integration tab
3. Select "Claude API" as provider
4. Enter your API key in the secure field
5. API key is saved encrypted locally

#### Setting Up OpenAI API  
1. Get your API key from https://platform.openai.com
2. Open Settings → AI Integration tab
3. Select "OpenAI API" as provider
4. Enter your API key in the secure field
5. Choose your preferred model (GPT-3.5 Turbo recommended for cost)
6. API key is saved encrypted locally

### AI Features

#### Auto-Send Mode
- **Automatic processing**: Every transcription is sent to AI immediately
- **Seamless workflow**: No manual intervention required
- **Real-time enhancement**: AI analysis appears alongside transcription
- **Enable**: Check "Automatically send transcriptions to AI" in settings

#### Manual Send Mode
- **On-demand processing**: Send transcriptions to AI when needed
- **Selective enhancement**: Choose which transcriptions to enhance
- **Cost control**: Only pay for API calls you make
- **Usage**: Click "Send to AI" button in the interface

### Downloading Models for Local AI

The application supports automatic model downloading:

#### TinyLlama (Recommended for Pi)
- **Size**: 638 MB
- **Performance**: Optimized for Raspberry Pi hardware
- **Download**: One-click download in AI settings
- **Location**: `~/simple-llm-server/models/`

#### Additional Models
For Phi-2 and Mistral models, use the LLM setup scripts:
```bash
# Navigate to LLM scripts
cd ~/llm-scripts/setup/

# Download and prepare models
./prepare_models.sh
```

### Example Use Cases

#### 1. Meeting Notes Enhancement
- Record meeting discussions with whisper transcription
- Auto-send to AI for summary and action items
- Export enhanced notes with timestamps

#### 2. Interview Transcription
- Transcribe interviews in real-time
- Send to AI for key points extraction
- Generate structured summaries

#### 3. Voice Memo Processing
- Record quick voice memos
- AI enhancement for clarity and structure
- Automatic organization and categorization

#### 4. Language Learning
- Practice pronunciation with transcription
- AI feedback on grammar and usage
- Multilingual support with AI translation

#### 5. Accessibility Support
- Real-time speech-to-text for accessibility
- AI enhancement for difficult audio
- Context-aware corrections

### Privacy & Security Considerations

#### Local AI
- ✅ **Complete Privacy**: No data leaves your device
- ✅ **Offline Capable**: Works without internet
- ✅ **No API Costs**: Free after model download
- ⚠️ **Storage Requirements**: Models require significant disk space

#### Cloud APIs (Claude/OpenAI)
- ⚠️ **Data Transmission**: Transcriptions sent to cloud services
- ✅ **High Quality**: Access to state-of-the-art models
- ⚠️ **API Costs**: Pay-per-use pricing
- ✅ **No Storage Required**: Models hosted remotely

## Features in Detail

### Pro Version Exclusive Features

#### 1. Modern UI with Themes
- CustomTkinter framework with smooth animations
- Dark and Light theme support
- System theme detection
- Live theme preview in settings

#### 2. Inline Microphone Testing
- Expandable test panel within settings
- 2-second test recording
- Real-time audio level meter
- Waveform visualization
- Playback functionality
- Audio quality analysis (avg/peak levels)

#### 3. AI-Powered Audio Enhancement (Hailo)
- Automatic Hailo detection
- Intelligent noise gate
- Automatic gain control
- Speech frequency emphasis
- Real-time processing

#### 4. Advanced Audio Processing
- High-pass Butterworth filtering for noise reduction
- Voice Activity Detection with dynamic thresholds
- Professional audio resampling
- Configurable block sizes

#### 5. Export Functionality
- Saves to ~/Documents/WhisperTranscriptions/
- Timestamped filenames
- Full transcription history
- Export confirmation with path

#### 6. Real-Time Visualization
- Live waveform display during recording
- Color-coded audio levels
- Smooth 50-sample rolling buffer
- Dynamic canvas updates

#### 7. Professional Status System
- Color-coded indicators:
  - Green: Ready/Success
  - Red: Recording/Error
  - Orange: Processing/Warning
  - Yellow: Notifications
- Auto-clearing notifications
- Comprehensive error messages

## Configuration

### Settings Storage
Pro version settings are stored in `~/.whisper_transcribe_pro.json` with automatic loading/saving.

### Audio Device Selection
Both versions automatically detect USB microphones. Pro version allows manual selection from all available devices.

### Model Selection (Pro Version)
- **tiny** (39MB) - Fastest, good for real-time
- **base** (74MB) - Better accuracy
- **small** (244MB) - Good balance
- **medium** (769MB) - Best accuracy

## Troubleshooting

### Audio Issues

```bash
# Check audio devices
arecord -l

# Test microphone (Pro version has built-in test)
arecord -D plughw:2,0 -f S16_LE -r 44100 -d 3 test.wav
aplay test.wav
```

### Application Won't Start

```bash
# Check if another instance is running
pkill -f whisper_transcribe_pro.py  # For Pro
pkill -f whisper_transcribe.py      # For Standard

# Reinstall dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### No Audio Detected

1. Use the Pro version's microphone test feature
2. Check USB microphone connection: `lsusb | grep -i audio`
3. Verify device in settings
4. Try different sample rates

### Settings Not Saving (Pro Version)

Check permissions for settings file:
```bash
ls -la ~/.whisper_transcribe_pro.json
```

## Performance

- **Raspberry Pi 5**: ~2-3 seconds per transcription
- **Raspberry Pi 4**: ~4-6 seconds per transcription
- **With Hailo**: Enhanced audio quality, same speed
- **Model Impact**: Larger models increase accuracy but slow transcription

## Project Structure

```
whisper-transcribe-pi/
├── whisper_transcribe.py          # Standard version
├── whisper_transcribe_pro.py      # Pro version with AI integration
├── launch_whisper.sh              # Standard launcher
├── launch_whisper_pro.sh          # Pro launcher
├── setup_pro.sh                   # Pro installation script
├── install.sh                     # Standard installation script
├── requirements.txt               # Python dependencies
├── whisper-transcribe.desktop     # Standard desktop entry
├── whisper-transcribe-pro.desktop # Pro desktop entry
├── ai_provider_usage_example.py   # AI integration example code
├── test_ai_integration.py         # AI integration testing
├── icons/                         # Application icons
│   ├── whisper-icon.png
│   └── whisper-icon.svg
├── PRO_FEATURES.md               # Detailed Pro features
├── MARKETING.md                  # Marketing documentation
├── CLAUDE.md                     # Development instructions
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Building from Source

```bash
# Clone repository
git clone https://github.com/sethshoultes/whisper-transcribe-pi.git
cd whisper-transcribe-pi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Pro version
python3 whisper_transcribe_pro.py

# Run Standard version
python3 whisper_transcribe.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern UI framework
- [SoundDevice](https://python-sounddevice.readthedocs.io/) for audio recording
- [Hailo](https://hailo.ai/) for AI acceleration support
- Built specifically for Raspberry Pi enthusiasts

## Author

Created for the Raspberry Pi community to enable professional speech-to-text functionality on Pi devices.

---

**Note**: The Pro version requires additional dependencies but provides a significantly enhanced user experience with professional features suitable for production use.