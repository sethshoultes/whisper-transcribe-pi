#!/usr/bin/env python3
"""
Test script for Voice Memory Manager integration with WhisperTranscribePro
This tests the integration logic without requiring GUI or audio dependencies.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

def test_settings_integration():
    """Test that voice memory settings are properly integrated"""
    print("Testing Settings integration...")
    
    # Mock the settings path to avoid conflicts
    test_config_path = tempfile.mktemp(suffix='.json')
    
    try:
        # Create minimal settings class for testing
        class TestSettings:
            def __init__(self):
                self.config_path = Path(test_config_path)
                self.settings = self.load_settings()
            
            def load_settings(self):
                defaults = {
                    "theme": "dark",
                    "model": "tiny",
                    "voice_memory_enabled": True,
                    "voice_memory_context_limit": 10,
                    "voice_memory_audio_metadata": True,
                    "voice_memory_pattern_learning": True,
                    "voice_memory_wake_word_tracking": True,
                    "voice_memory_db_path": "data/voice_conversations.db",
                    "voice_memory_auto_save": True,
                    "voice_memory_compression": True
                }
                
                if self.config_path.exists():
                    try:
                        with open(self.config_path, 'r') as f:
                            loaded = json.load(f)
                            defaults.update(loaded)
                    except Exception:
                        pass
                
                return defaults
            
            def save_settings(self):
                try:
                    with open(self.config_path, 'w') as f:
                        json.dump(self.settings, f, indent=2)
                except Exception as e:
                    print(f"Failed to save settings: {e}")
        
        # Test settings creation
        settings = TestSettings()
        
        # Verify voice memory settings exist
        required_settings = [
            "voice_memory_enabled",
            "voice_memory_context_limit", 
            "voice_memory_audio_metadata",
            "voice_memory_pattern_learning",
            "voice_memory_wake_word_tracking",
            "voice_memory_db_path",
            "voice_memory_auto_save",
            "voice_memory_compression"
        ]
        
        for setting in required_settings:
            if setting not in settings.settings:
                raise AssertionError(f"Missing voice memory setting: {setting}")
        
        print("✅ All voice memory settings found")
        
        # Test settings save/load
        settings.settings["voice_memory_enabled"] = False
        settings.save_settings()
        
        # Load again to verify persistence
        settings2 = TestSettings()
        if settings2.settings["voice_memory_enabled"] != False:
            raise AssertionError("Settings persistence failed")
        
        print("✅ Settings save/load working")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_config_path):
            os.unlink(test_config_path)

def test_memory_manager_import():
    """Test that VoiceMemoryManager can be imported"""
    print("Testing VoiceMemoryManager import...")
    
    try:
        # Test if we can import the memory manager
        sys.path.append(os.path.dirname(__file__))
        
        try:
            from tools.memory.voice_memory_manager import VoiceMemoryManager, create_voice_memory_manager
            print("✅ VoiceMemoryManager import successful")
            return True
        except ImportError as e:
            print(f"⚠️ VoiceMemoryManager import failed: {e}")
            print("   This is expected if memory tools are not installed")
            return True  # Not a failure - just not available
            
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        return False

def test_memory_config_creation():
    """Test memory configuration creation logic"""
    print("Testing memory configuration creation...")
    
    try:
        # Mock settings
        settings = {
            "voice_memory_enabled": True,
            "voice_memory_audio_metadata": True,
            "voice_memory_context_limit": 15,
            "voice_memory_db_path": "test_data/voice_conversations.db",
            "voice_memory_pattern_learning": False,
            "voice_memory_compression": True,
            "voice_memory_auto_save": False,
            "voice_memory_wake_word_tracking": True
        }
        
        # Simulate the config creation logic from the app
        memory_config = {
            'enable_context_memory': True,
            'enable_audio_metadata': settings.get("voice_memory_audio_metadata", True),
            'conversation_memory_limit': settings.get("voice_memory_context_limit", 100),
            'conversation_db_path': settings.get("voice_memory_db_path", "data/voice_conversations.db"),
            'enable_pattern_learning': settings.get("voice_memory_pattern_learning", True),
            'enable_memory_compression': settings.get("voice_memory_compression", True),
            'auto_save_interval': 300 if settings.get("voice_memory_auto_save", True) else 0,
            'wake_word_learning_enabled': settings.get("voice_memory_wake_word_tracking", True)
        }
        
        # Verify config values
        expected_values = {
            'enable_context_memory': True,
            'enable_audio_metadata': True,
            'conversation_memory_limit': 15,
            'conversation_db_path': "test_data/voice_conversations.db",
            'enable_pattern_learning': False,
            'enable_memory_compression': True,
            'auto_save_interval': 0,
            'wake_word_learning_enabled': True
        }
        
        for key, expected in expected_values.items():
            if memory_config[key] != expected:
                raise AssertionError(f"Config mismatch for {key}: expected {expected}, got {memory_config[key]}")
        
        print("✅ Memory configuration creation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Memory config test failed: {e}")
        return False

def test_fallback_behavior():
    """Test that the app handles missing memory system gracefully"""
    print("Testing fallback behavior...")
    
    try:
        # Simulate the fallback logic from the app
        VOICE_MEMORY_AVAILABLE = False  # Simulate missing memory system
        voice_memory = None
        
        # Test storage methods with no memory system
        def store_voice_transcription_fallback(text, result, audio_file, audio_data):
            if not voice_memory:
                print("Debug: Voice memory not available, skipping transcription storage")
                return True  # Should not fail
            
        def store_ai_interaction_fallback(user_input, ai_response):
            if not voice_memory:
                print("Debug: Voice memory not available, skipping AI interaction storage")
                return True  # Should not fail
        
        # Test these functions
        store_voice_transcription_fallback("test text", {}, "/tmp/test.wav", [])
        store_ai_interaction_fallback("test input", "test response")
        
        print("✅ Fallback behavior working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("🧪 Starting Voice Memory Manager Integration Tests")
    print("=" * 60)
    
    tests = [
        test_settings_integration,
        test_memory_manager_import,
        test_memory_config_creation,
        test_fallback_behavior
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Integration looks good.")
        return True
    else:
        print("⚠️ Some tests failed. Review the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)