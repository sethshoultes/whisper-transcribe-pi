#!/usr/bin/env python3
"""Quick test of OpenAI integration"""

# Test if we can create an OpenAI provider
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the OpenAI provider from our main file
import whisper_transcribe_pro

# Test with a dummy API key
test_key = "sk-test123"  # This will fail but we can see if the provider initializes

provider = whisper_transcribe_pro.OpenAIProvider(api_key=test_key)
print(f"Provider created: {provider.get_name()}")
print(f"Provider status: {provider.get_status()}")
print(f"Is available: {provider.is_available()}")

# Now test if it correctly detects missing library or invalid key
if not provider.is_available():
    print("\nAs expected, provider is not available without valid key")
    print("This shows the error handling is working correctly")