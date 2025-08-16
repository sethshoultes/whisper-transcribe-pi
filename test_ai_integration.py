#!/usr/bin/env python3
"""Comprehensive test of AI integration with focus on OpenAI"""

import json
import sys
import os
import threading
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_settings():
    """Test that settings are properly configured for OpenAI"""
    print("=" * 50)
    print("TESTING SETTINGS")
    print("=" * 50)
    
    settings_file = os.path.expanduser("~/.whisper_transcribe_pro.json")
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    
    print(f"AI Provider: {settings.get('ai_provider')}")
    print(f"AI Enabled: {settings.get('ai_enabled')}")
    print(f"OpenAI Model: {settings.get('openai_model')}")
    print(f"OpenAI API Key: {'Present' if settings.get('openai_api_key') else 'Missing'}")
    
    assert settings.get('ai_provider') == 'openai', "Provider should be 'openai'"
    assert settings.get('ai_enabled') == True, "AI should be enabled"
    assert settings.get('openai_api_key'), "OpenAI API key should be present"
    
    print("✓ Settings are correct\n")
    return settings

def test_provider_creation(settings):
    """Test that OpenAI provider can be created"""
    print("=" * 50)
    print("TESTING PROVIDER CREATION")
    print("=" * 50)
    
    from whisper_transcribe_pro import OpenAIProvider
    
    api_key = settings.get('openai_api_key')
    model = settings.get('openai_model', 'gpt-3.5-turbo')
    
    provider = OpenAIProvider(api_key=api_key, model=model)
    print(f"Provider Name: {provider.get_name()}")
    print(f"Provider Status: {provider.get_status()}")
    print(f"Is Available: {provider.is_available()}")
    
    assert provider.is_available(), "Provider should be available with API key"
    print("✓ Provider created successfully\n")
    return provider

def test_ui_label():
    """Test what the UI label would show"""
    print("=" * 50)
    print("TESTING UI LABEL DISPLAY")
    print("=" * 50)
    
    # Load settings directly from file without creating Settings instance
    settings_file = os.path.expanduser("~/.whisper_transcribe_pro.json")
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    
    provider_type = settings.get("ai_provider", "local")
    ai_enabled = settings.get("ai_enabled", False)
    
    if not ai_enabled:
        label_text = "AI: Disabled"
        label_color = "gray"
    elif provider_type == "openai":
        api_key = settings.get("openai_api_key", "")
        if api_key:
            from whisper_transcribe_pro import OpenAIProvider
            provider = OpenAIProvider(api_key=api_key)
            if provider.is_available():
                model = settings.get("openai_model", "gpt-3.5-turbo")
                model_short = model.replace("gpt-", "GPT-")
                label_text = f"AI: {model_short}"
                label_color = "green"
            else:
                label_text = "AI: OpenAI (Not Available)"
                label_color = "red"
        else:
            label_text = "AI: OpenAI (No API Key)"
            label_color = "red"
    else:
        label_text = f"AI: {provider_type}"
        label_color = "gray"
    
    print(f"Label Text: {label_text}")
    print(f"Label Color: {label_color}")
    
    assert "GPT" in label_text, f"Label should show GPT model, got: {label_text}"
    assert label_color == "green", f"Label should be green, got: {label_color}"
    print("✓ UI label would display correctly\n")

def test_message_sending(provider):
    """Test sending a message to OpenAI"""
    print("=" * 50)
    print("TESTING MESSAGE SENDING")
    print("=" * 50)
    
    test_message = "Say 'Hello from OpenAI' and nothing else."
    print(f"Sending: {test_message}")
    
    try:
        response = provider.send_message(test_message)
        print(f"Response: {response}")
        
        if response and "error" not in response.lower():
            print("✓ Message sent successfully\n")
        else:
            print("⚠ Response received but may contain error\n")
    except Exception as e:
        print(f"✗ Error sending message: {e}\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("AI INTEGRATION TEST - OPENAI FOCUS")
    print("=" * 50 + "\n")
    
    try:
        # Test 1: Check settings
        settings = test_settings()
        
        # Test 2: Create provider
        provider = test_provider_creation(settings)
        
        # Test 3: Check UI label
        test_ui_label()
        
        # Test 4: Send test message (optional - costs API credits)
        print("=" * 50)
        print("OPTIONAL: TEST MESSAGE SENDING")
        print("=" * 50)
        print("Skipping message test to avoid API costs")
        print("Uncomment test_message_sending(provider) to test\n")
        # test_message_sending(provider)
        
        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        print("\nThe app should now show 'AI: GPT-3.5-turbo' at the bottom")
        print("when you run ./launch_whisper_pro.sh")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()