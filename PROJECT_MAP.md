# Whisper Transcribe Pi - Project Map

## Project Structure
```
/home/sethshoultes/whisper-transcribe-pi/
│
├── Main Applications
│   ├── whisper_transcribe.py         # Standard version (tkinter, lightweight)
│   └── whisper_transcribe_pro.py     # Pro version (customtkinter, full-featured)
│
├── Launcher Scripts
│   ├── launch_whisper.sh             # Standard launcher (activates venv)
│   └── launch_whisper_pro.sh         # Pro launcher (nohup for background)
│
├── Installation & Setup
│   ├── install.sh                    # Standard installation script
│   ├── setup_pro.sh                  # Pro installation with dependencies
│   └── requirements.txt              # Python dependencies (whisper, customtkinter, etc.)
│
├── Desktop Integration
│   ├── whisper-transcribe.desktop    # Standard desktop entry
│   └── whisper-transcribe-pro.desktop # Pro desktop entry
│
├── Documentation
│   ├── README.md                     # Main documentation
│   ├── CLAUDE.md                     # Development instructions
│   ├── PROJECT_MAP.md                # This file - project structure map
│   ├── PRO_FEATURES.md              # Pro features detailed
│   └── MARKETING.md                  # Marketing strategy
│
└── Icons
    └── icons/create_icon.py          # Icon generator script
```

## Key Features Map

### Standard Version (`whisper_transcribe.py`)
- **GUI**: Simple tkinter interface
- **Recording**: Click to record → Stop → Transcribe
- **Clipboard**: Auto-copy transcriptions
- **Shortcuts**: R/Space to record

### Pro Version (`whisper_transcribe_pro.py`) - Line References
- **Modern UI** (Lines 1-100): CustomTkinter setup
- **Settings Window** (Lines 842-1594): 
  - `class SettingsWindow` - Comprehensive 4-tab interface
  - Audio Tab (Lines 940-1061): Device selection, inline test panel
  - Transcription Tab (Lines 1110-1169): Model & language selection
  - Interface Tab (Lines 1170-1260): Themes, window size, fonts
  - Advanced Tab (Lines 1261-1330): Hotkeys, debug settings
  
- **Inline Microphone Test** (Lines 1359-1513):
  - `toggle_test_microphone()`: Show/hide test panel
  - `start_test_recording()`: 2-second recording
  - `playback_test_recording()`: Audio playback
  - `draw_test_waveform()`: Mini waveform visualization

- **Audio Processing** (Lines 210-300):
  - Noise reduction with Butterworth filter
  - Voice Activity Detection (VAD)
  - Audio resampling (44.1kHz → 16kHz)

- **Hailo Integration** (Lines 55-130):
  - `class HailoIntegration`: AI audio enhancement
  - Noise gate, AGC, speech emphasis

- **Real-time Visualization** (Lines 685-740):
  - Waveform canvas during recording
  - Color-coded audio levels
  - 50-sample rolling buffer

- **Export System** (Lines 620-650):
  - `export_transcription()`: Save to ~/Documents/WhisperTranscriptions/
  - Timestamped filenames

## Current State (master branch)
- [DONE] Inline microphone test (no modal window)
- [DONE] Fixed emoji display issues for Pi
- [DONE] Settings apply immediately
- [DONE] Hailo AI audio enhancement
- [DONE] Comprehensive README updated
- [REMOVED] Universal version removed (not functional)

## Dependencies
- `openai-whisper`: Speech recognition
- `customtkinter`: Modern UI (Pro only)
- `sounddevice`: Audio recording
- `scipy`: Audio processing
- `numpy`: Array operations
- `pyaudio`: Audio I/O

## Quick Commands
```bash
# Run Pro version
./launch_whisper_pro.sh

# Run Standard version  
./launch_whisper.sh

# Install Pro
./setup_pro.sh

# Settings location
~/.whisper_transcribe_pro.json

# Export location
~/Documents/WhisperTranscriptions/

# Logs location
/tmp/whisper_pro.log
```

## Key Integration Points

### Audio Flow
1. **Input**: USB Microphone (44.1kHz) -> sounddevice
2. **Processing**: Resampling -> Noise Reduction -> VAD -> Hailo Enhancement
3. **Transcription**: Whisper Model (tiny/base/small/medium)
4. **Output**: Text Display -> Clipboard -> Export File

### Settings Management
- **Storage**: `~/.whisper_transcribe_pro.json`
- **Hot Reload**: All settings apply immediately
- **Defaults**: Fallback values for missing/corrupt settings

### UI Components
- **Main Window**: Recording button, transcription display, action buttons
- **Settings Window**: 4-tab modal with scrollable frames
- **Test Panel**: Inline expandable microphone test
- **Status Bar**: Model, Hailo status, notifications

## Common Issues & Solutions

| Issue | Solution | File/Line |
|-------|----------|-----------|
| No audio detected | Check USB mic, use inline test | Lines 1359-1513 |
| Settings not saving | Check ~/.whisper_transcribe_pro.json permissions | Line 140 |
| Emoji display issues | Fixed - using ASCII alternatives | Throughout |
| Window opacity not working | Removed - Wayland incompatible | N/A |
| Model switching slow | Background loading implemented | Lines 1140-1160 |

## Performance Metrics
- **Startup**: ~2-3 seconds (model loading)
- **Recording**: Real-time with 512 sample blocks
- **Transcription**: 2-3s (Pi5), 4-6s (Pi4)
- **UI Updates**: 50ms polling for smooth display
- **Memory**: ~200MB (tiny model), ~800MB (medium model)