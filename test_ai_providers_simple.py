#!/usr/bin/env python3
"""
Simple test for AIProvider classes without full dependencies
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

# Recreate just the AI provider classes for testing
class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def send_message(self, text: str) -> str:
        """Send text message and get response"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is ready and available"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return provider name"""
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Return current status description"""
        pass

class ClaudeAPIProvider(AIProvider):
    """Claude API provider using Anthropic's API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = "claude-3-haiku-20240307"
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the Anthropic client"""
        if not self.api_key:
            print("Claude API key not found. Set ANTHROPIC_API_KEY environment variable.")
            return
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("Claude API client initialized successfully")
        except ImportError:
            print("anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            print(f"Failed to initialize Claude API client: {e}")
    
    def send_message(self, text: str) -> str:
        """Send message to Claude API"""
        if not self.client:
            return "Error: Claude API not available. Check API key and internet connection."
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            return response.content[0].text
        
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                return "Error: Rate limit exceeded. Please wait before trying again."
            elif "invalid api key" in error_msg or "authentication" in error_msg:
                return "Error: Invalid API key. Please check your Anthropic API key."
            elif "network" in error_msg or "connection" in error_msg:
                return "Error: Network connection failed. Check your internet connection."
            else:
                return f"Error: Claude API request failed - {e}"
    
    def is_available(self) -> bool:
        """Check if Claude API is available"""
        return self.client is not None and self.api_key is not None
    
    def get_name(self) -> str:
        """Return provider name"""
        return "Claude API (Haiku)"
    
    def get_status(self) -> str:
        """Return current status"""
        if not self.api_key:
            return "Missing API key"
        elif not self.client:
            return "Client setup failed"
        else:
            return "Ready"

class OpenAIProvider(AIProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the OpenAI client"""
        if not self.api_key:
            print("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            print(f"OpenAI API client initialized successfully with model {self.model}")
        except ImportError:
            print("openai library not installed. Run: pip install openai")
        except Exception as e:
            print(f"Failed to initialize OpenAI API client: {e}")
    
    def send_message(self, text: str) -> str:
        """Send message to OpenAI API"""
        if not self.client:
            return "Error: OpenAI API not available. Check API key and internet connection."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": text}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                return "Error: Rate limit exceeded. Please wait before trying again."
            elif "invalid api key" in error_msg or "authentication" in error_msg:
                return "Error: Invalid API key. Please check your OpenAI API key."
            elif "network" in error_msg or "connection" in error_msg:
                return "Error: Network connection failed. Check your internet connection."
            elif "model" in error_msg and "not found" in error_msg:
                return f"Error: Model {self.model} not available. Try gpt-3.5-turbo or gpt-4."
            else:
                return f"Error: OpenAI API request failed - {e}"
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        return self.client is not None and self.api_key is not None
    
    def get_name(self) -> str:
        """Return provider name"""
        return f"OpenAI ({self.model})"
    
    def get_status(self) -> str:
        """Return current status"""
        if not self.api_key:
            return "Missing API key"
        elif not self.client:
            return "Client setup failed"
        else:
            return f"Ready ({self.model})"
    
    def set_model(self, model: str):
        """Change the model used by this provider"""
        self.model = model
        print(f"OpenAI model changed to {model}")

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
    
    # Test OpenAIProvider
    print("\n2. Testing OpenAIProvider:")
    openai_provider = OpenAIProvider()
    print(f"   Name: {openai_provider.get_name()}")
    print(f"   Status: {openai_provider.get_status()}")
    print(f"   Available: {openai_provider.is_available()}")
    
    # Test model switching for OpenAI
    print("\n   Testing model switching:")
    openai_provider.set_model("gpt-4")
    print(f"   After switching to GPT-4: {openai_provider.get_name()}")
    
    print("\n" + "=" * 50)
    print("Basic functionality test completed!")
    
    # Check environment variables
    print("\nEnvironment check:")
    print(f"ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    
    print("\nNote: To test actual API calls, set the appropriate API keys and run again.")

if __name__ == "__main__":
    test_ai_providers()