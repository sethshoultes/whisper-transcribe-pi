# OpenAI Provider Selection Fix Summary

## Issue
User reported that despite selecting OpenAI in settings and adding an API key, the app still showed "TinyLlama" in the status bar at the bottom.

## Root Cause
The OpenAI model selection dropdown didn't have a trace callback to save the selection immediately when changed, unlike the API key field.

## Fix Applied
Added a trace callback to the OpenAI model dropdown to save the selection immediately:

```python
# Line 2167-2168 in whisper_transcribe_pro.py
self.openai_model_combo.pack(padx=10, pady=(2,10))
# Save model selection when it changes
self.openai_model_var.trace("w", lambda *args: self.save_api_keys())
```

## Verification
1. Settings file now correctly stores:
   - `"ai_provider": "openai"`
   - `"openai_model": "gpt-3.5-turbo"`
   - OpenAI API key

2. Test script confirms:
   - Provider creation works
   - Settings persist correctly
   - UI label would show "AI: GPT-3.5-turbo"

## Status Bar Display
The app will now show the correct status based on the selected provider:
- **Local AI**: Shows model name (e.g., "AI: TinyLlama")
- **OpenAI**: Shows "AI: GPT-3.5-turbo" (or selected model)
- **Claude**: Shows "AI: Claude API"

## How to Use
1. Open Settings → AI Integration tab
2. Select "OpenAI API (GPT models)"
3. Enter your OpenAI API key
4. Select desired model from dropdown
5. Enable AI features
6. Close settings

The status bar at the bottom will now correctly show "AI: GPT-3.5-turbo" instead of TinyLlama.