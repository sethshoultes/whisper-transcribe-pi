#!/usr/bin/env python3
"""
Test script for AIProvider implementations
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path to import our module
sys.path.insert(0, str(Path(__file__).parent))

from whisper_transcribe_pro import AIProvider, ClaudeAPIProvider, OpenAIProvider, LocalAI, Settings

def test_ai_providers():
    """Test all AI provider implementations"""
    print("Testing AI Provider implementations...")
    print("=" * 50)
    
    # Test ClaudeAPIProvider
    print("\n1. Testing ClaudeAPIProvider:")
    claude = ClaudeAPIProvider()
    print(f"   Name: {claude.get_name()}")
    print(f"   Status: {claude.get_status()}")
    print(f"   Available: {claude.is_available()}")
    
    if claude.is_available():
        response = claude.send_message("Hello, this is a test message.")
        print(f"   Test response: {response[:100]}...")
    else:
        print("   Skipping message test (API key not available)")
    
    # Test OpenAIProvider
    print("\n2. Testing OpenAIProvider:")
    openai_provider = OpenAIProvider()
    print(f"   Name: {openai_provider.get_name()}")
    print(f"   Status: {openai_provider.get_status()}")
    print(f"   Available: {openai_provider.is_available()}")
    
    if openai_provider.is_available():
        response = openai_provider.send_message("Hello, this is a test message.")
        print(f"   Test response: {response[:100]}...")
    else:
        print("   Skipping message test (API key not available)")
    
    # Test model switching for OpenAI
    print("\n   Testing model switching:")
    openai_provider.set_model("gpt-4")
    print(f"   After switching to GPT-4: {openai_provider.get_name()}")
    
    # Test LocalAI
    print("\n3. Testing LocalAI:")
    settings = Settings()
    local_ai = LocalAI(settings)
    print(f"   Name: {local_ai.get_name()}")
    print(f"   Status: {local_ai.get_status()}")
    print(f"   Available: {local_ai.is_available()}")
    print(f"   Enabled in settings: {local_ai.is_enabled()}")
    print(f"   Current model: {local_ai.get_current_model()}")
    
    if local_ai.is_available():
        response = local_ai.send_message("Hello, this is a test message.")
        print(f"   Test response: {response[:100]}...")
    else:
        print("   Skipping message test (server not running)")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    # Check environment variables
    print("\nEnvironment check:")
    print(f"ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")

if __name__ == "__main__":
    test_ai_providers()