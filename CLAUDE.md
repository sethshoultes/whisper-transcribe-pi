# CLAUDE.md

Development guide for Claude Code when working with Whisper Transcribe Pi.

## Navigation
See `PROJECT_MAP.md` for complete project structure and line references.

## Quick Start

### Run Commands
```bash
# Pro version
./launch_whisper_pro.sh

# Standard version
./launch_whisper.sh

# Install dependencies
./setup_pro.sh  # or ./install.sh for standard
```

### Key Locations
- Settings: `~/.whisper_transcribe_pro.json`
- Exports: `~/Documents/WhisperTranscriptions/`
- Logs: `/tmp/whisper_pro.log`

## Development Guidelines

### Code Style
- 4-space indentation
- Snake_case for functions/variables
- CamelCase for classes
- Use type hints where possible
- Handle exceptions with specific messages

### Testing Changes
```bash
# Quick test
source venv/bin/activate
python3 whisper_transcribe_pro.py

# Check audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
arecord -D plughw:2,0 -f S16_LE -r 44100 -d 3 test.wav && aplay test.wav
```

## Common Tasks

### Debugging Audio Issues
1. Check USB mic detection in `detect_usb_microphone()` (Line ~250)
2. Verify sample rate handling in `transcribe_audio()` (Line ~300)
3. Test with inline microphone test panel (Lines 1359-1513)

### Modifying Settings
1. Update `SettingsWindow` class (Lines 842-1594)
2. Add to settings dict in `Settings` class (Line ~140)
3. Implement `apply_settings()` for immediate application

### Adding Features
1. Pro features go in `whisper_transcribe_pro.py`
2. Update `PROJECT_MAP.md` with line references
3. Document in README.md user-facing changes
4. Test on actual Raspberry Pi hardware

## Critical Rules

### NO Mock Data
- Always use REAL audio devices
- Show REAL error messages
- No fallback responses that hide problems
- Test with actual hardware

### Platform Specifics
- Target Raspberry Pi 4/5 with USB microphones
- Handle 44.1kHz → 16kHz resampling
- ASCII only (no emoji) for Pi compatibility
- Consider Wayland limitations (no window opacity)

## Known Issues

| Issue | Status | Solution |
|-------|--------|----------|
| Emoji display on Pi | Fixed | Using ASCII alternatives |
| Window opacity | Removed | Wayland incompatible |
| Settings not applying | Fixed | Immediate application implemented |
| Modal test window | Fixed | Inline panel instead |

## Git Workflow

```bash
# Feature branch
git checkout -b feature/new-feature

# Commit with clear message
git add -A
git commit -m "Add feature: description"

# Push and create PR
git push -u origin feature/new-feature
```

## Quick Reference

### Key Classes
- `WhisperTranscribePro` - Main application (Line 132)
- `SettingsWindow` - Settings interface (Line 842) 
- `HailoIntegration` - AI enhancement (Line 55)
- `Settings` - Config management (Line 140)

### Important Methods
- `record_audio()` - Audio capture (Line ~400)
- `transcribe_audio()` - Whisper processing (Line ~450)
- `export_transcription()` - Save to file (Line ~620)
- `toggle_test_microphone()` - Inline test (Line 1359)

See `PROJECT_MAP.md` for complete line references and component details.