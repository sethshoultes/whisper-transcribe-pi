#!/usr/bin/env python3
"""Test script to verify OpenAI status display"""

import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load settings
settings_file = os.path.expanduser("~/.whisper_transcribe_pro.json")
with open(settings_file, 'r') as f:
    settings = json.load(f)

print("Current Settings:")
print(f"  AI Provider: {settings.get('ai_provider', 'not set')}")
print(f"  AI Enabled: {settings.get('ai_enabled', False)}")
print(f"  OpenAI Model: {settings.get('openai_model', 'not set')}")
print(f"  OpenAI Key: {'Set' if settings.get('openai_api_key') else 'Not set'}")
print()

# Import relevant parts from main app
from whisper_transcribe_pro import Settings, OpenAIProvider

# Create settings object
app_settings = Settings()

# Check what the app would show
provider_type = app_settings.settings.get("ai_provider", "local")
ai_enabled = app_settings.settings.get("ai_enabled", False)

print("App Status Display:")
if not ai_enabled:
    print("  AI: Disabled")
elif provider_type == "openai":
    api_key = app_settings.settings.get("openai_api_key", "")
    if api_key:
        provider = OpenAIProvider(api_key=api_key)
        if provider.is_available():
            model = app_settings.settings.get("openai_model", "gpt-3.5-turbo")
            model_short = model.replace("gpt-", "GPT-")
            print(f"  AI: {model_short} (OpenAI)")
        else:
            print("  AI: OpenAI (Not Available)")
    else:
        print("  AI: OpenAI (No API Key)")
elif provider_type == "local":
    print("  AI: Local (would show model if server running)")
elif provider_type == "claude":
    print("  AI: Claude")