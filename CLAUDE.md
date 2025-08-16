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

### Testing AI Providers
```bash
# Test OpenAI integration
python3 -c "from whisper_transcribe_pro import test_openai_connection; test_openai_connection()"

# Test Anthropic Claude integration  
python3 -c "from whisper_transcribe_pro import test_anthropic_connection; test_anthropic_connection()"

# Test Groq integration
python3 -c "from whisper_transcribe_pro import test_groq_connection; test_groq_connection()"

# Full AI integration test
python3 -c "from whisper_transcribe_pro import WhisperTranscribePro; app = WhisperTranscribePro(); app.test_ai_integration()"
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

### AI Integration Tasks
1. Test AI providers with `test_ai_integration()` (Line ~700)
2. Configure provider settings in AI tab (Lines 1150-1200)
3. Monitor AI enhancement status in main UI
4. Debug provider responses in console output
5. Validate API keys and connection status

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

## AI Integration Issues & Solutions

### Debugging AI Integration
```bash
# Check AI provider status
tail -f /tmp/whisper_pro.log | grep -i "ai\|provider\|enhancement"

# Test provider connectivity
python3 -c "import requests; print('Network OK' if requests.get('https://api.openai.com', timeout=5).status_code else 'Network Issue')"

# Validate API key format
python3 -c "import os; key = os.getenv('OPENAI_API_KEY'); print('Valid key format' if key and key.startswith('sk-') else 'Invalid key')"

# Debug enhancement process
python3 -c "from whisper_transcribe_pro import WhisperTranscribePro; app = WhisperTranscribePro(); app.debug_enhancement_process()"
```

### Common AI Issues

| Issue | Cause | Solution |
|-------|--------|----------|
| API key invalid | Wrong format/expired | Check key format in settings (Line 1175) |
| Network timeout | Connection issues | Increase timeout in `enhance_with_ai()` (Line ~750) |
| Rate limiting | Too many requests | Implement backoff in provider classes (Line ~800) |
| Empty responses | Provider error | Check error handling in `call_ai_provider()` (Line ~850) |
| Enhancement failed | Model unavailable | Fallback logic in `enhance_transcription()` (Line ~780) |

### Important AI Code Sections

#### Core AI Integration
- `HailoIntegration` class initialization (Lines 55-85)
- AI enhancement toggle (Lines 720-740)
- Provider selection logic (Lines 760-790)
- Error handling for AI calls (Lines 810-840)

#### Provider Implementations
- OpenAI integration (Lines 870-920)
- Anthropic Claude integration (Lines 930-980)
- Groq integration (Lines 990-1040)
- Provider testing methods (Lines 1050-1100)

#### Settings & UI
- AI settings tab (Lines 1150-1200)
- Provider configuration (Lines 1210-1260)
- API key management (Lines 1270-1300)
- Enhancement status display (Lines 1310-1340)

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

### AI Integration Methods
- `enhance_with_ai()` - Core AI enhancement (Line ~750)
- `call_ai_provider()` - Provider interface (Line ~850)
- `test_ai_integration()` - Connectivity test (Line ~700)
- `setup_ai_providers()` - Provider initialization (Line ~680)
- `handle_ai_error()` - Error management (Line ~900)

See `PROJECT_MAP.md` for complete line references and component details.