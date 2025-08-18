#!/usr/bin/env python3
"""
Example usage of AIProvider classes
This shows how the providers can be used interchangeably
"""

import os
from typing import List

# In a real application, these would be imported from whisper_transcribe_pro
# from whisper_transcribe_pro import AIProvider, ClaudeAPIProvider, OpenAIProvider, LocalAI

def example_ai_manager():
    """Example of managing multiple AI providers"""
    
    print("AI Provider Manager Example")
    print("=" * 40)
    
    # This would be how you'd initialize providers in the main application
    providers = []
    
    # Add Claude if API key is available
    if os.getenv('ANTHROPIC_API_KEY'):
        from whisper_transcribe_pro import ClaudeAPIProvider
        claude = ClaudeAPIProvider()
        providers.append(claude)
        print(f"✓ Added: {claude.get_name()}")
    
    # Add OpenAI if API key is available
    if os.getenv('OPENAI_API_KEY'):
        from whisper_transcribe_pro import OpenAIProvider
        openai_provider = OpenAIProvider()
        providers.append(openai_provider)
        print(f"✓ Added: {openai_provider.get_name()}")
        
        # You could also add multiple OpenAI providers with different models
        gpt4_provider = OpenAIProvider(model="gpt-4")
        providers.append(gpt4_provider)
        print(f"✓ Added: {gpt4_provider.get_name()}")
    
    # Add Local AI (would check if server is running)
    try:
        from whisper_transcribe_pro import LocalAI, Settings
        settings = Settings()
        local_ai = LocalAI(settings)
        providers.append(local_ai)
        print(f"✓ Added: {local_ai.get_name()}")
    except:
        print("✗ Could not load LocalAI (dependencies missing)")
    
    if not providers:
        print("No providers available. Set API keys or start local AI server.")
        return
    
    print(f"\nTotal providers: {len(providers)}")
    
    # Show status of all providers
    print("\nProvider Status:")
    print("-" * 30)
    for provider in providers:
        status_icon = "✓" if provider.is_available() else "✗"
        print(f"{status_icon} {provider.get_name()}: {provider.get_status()}")
    
    # Find available providers
    available_providers = [p for p in providers if p.is_available()]
    
    if available_providers:
        print(f"\nReady to use: {len(available_providers)} providers")
        
        # Example of sending a message to the first available provider
        test_message = "Hello! Can you help me understand what a whisper transcription app does?"
        print(f"\nTest message: '{test_message}'")
        print("-" * 50)
        
        provider = available_providers[0]
        print(f"Using: {provider.get_name()}")
        response = provider.send_message(test_message)
        print(f"Response: {response}")
        
    else:
        print("\nNo providers are currently available.")
        print("Set API keys or start local AI server to test messaging.")

def provider_selection_example():
    """Example of how a UI might let users select providers"""
    
    print("\n" + "=" * 40)
    print("Provider Selection Example")
    print("=" * 40)
    
    # This would be used in a dropdown or settings menu
    provider_configs = [
        {"name": "Claude Haiku (Fast & Cheap)", "class": "ClaudeAPIProvider", "requires": "ANTHROPIC_API_KEY"},
        {"name": "GPT-3.5 Turbo", "class": "OpenAIProvider", "model": "gpt-3.5-turbo", "requires": "OPENAI_API_KEY"},
        {"name": "GPT-4 (Premium)", "class": "OpenAIProvider", "model": "gpt-4", "requires": "OPENAI_API_KEY"},
        {"name": "Local AI (Private)", "class": "LocalAI", "requires": "Local server running"},
    ]
    
    print("Available AI Provider Options:")
    for i, config in enumerate(provider_configs):
        print(f"{i+1}. {config['name']} - Requires: {config['requires']}")
    
    print("\nIn the UI, users would select from these options,")
    print("and the app would initialize the appropriate provider.")

if __name__ == "__main__":
    example_ai_manager()
    provider_selection_example()