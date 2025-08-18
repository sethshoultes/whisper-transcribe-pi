#!/usr/bin/env python3
"""Debug memory system to understand conversation loading issues"""

import logging
import sys
from pathlib import Path

# Add tools to path
sys.path.append(str(Path(__file__).parent / "tools"))

from tools.memory.voice_memory_manager import VoiceMemoryManager

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def test_memory_loading():
    """Test memory loading to debug dialog issues"""
    try:
        print("Initializing Voice Memory Manager...")
        memory_manager = VoiceMemoryManager()
        print("✅ Memory manager initialized")
        
        print("\nTesting conversation memory get_recent...")
        recent = memory_manager.conversation_memory.get_recent(limit=5, session_only=False)
        print(f"Found {len(recent)} recent conversations")
        
        for i, conv in enumerate(recent):
            print(f"\nConversation {i+1}:")
            print(f"  Timestamp: {conv.get('timestamp', 'N/A')}")
            print(f"  User: {conv.get('user', 'N/A')}")
            print(f"  Assistant: {conv.get('assistant', 'N/A')}")
            print(f"  Keys available: {list(conv.keys())}")
        
        print("\nTesting context memory...")
        if hasattr(memory_manager.context_memory, 'conversation_history'):
            context_conversations = memory_manager.context_memory.conversation_history
            print(f"Found {len(context_conversations)} context conversations")
            
            for i, conv in enumerate(context_conversations[-3:]):
                print(f"\nContext conversation {i+1}:")
                print(f"  Timestamp: {conv.get('timestamp', 'N/A')}")
                print(f"  User: {conv.get('user', 'N/A')}")
                print(f"  Assistant: {conv.get('assistant', 'N/A')}")
        else:
            print("No conversation_history attribute in context memory")
            
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_loading()