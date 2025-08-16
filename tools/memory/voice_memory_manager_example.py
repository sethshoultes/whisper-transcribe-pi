#!/usr/bin/env python3
"""
Voice Memory Manager Usage Example

This example demonstrates how to use the VoiceMemoryManager as the central
coordination point for all voice memory operations in the whisper-transcribe-pi project.

The VoiceMemoryManager provides:
- Unified interface for both VoiceContextMemory and VoiceConversationMemory
- Memory fusion for comprehensive context
- Pattern analysis and learning
- Voice interaction lifecycle management
- Analytics and reporting
"""

import logging
import time
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.memory import VoiceMemoryManager, create_voice_memory_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate VoiceMemoryManager usage"""
    
    print("🧠🎤 Voice Memory Manager Usage Example")
    print("=" * 60)
    
    # 1. Create and configure the memory manager
    print("\n1. Initializing Voice Memory Manager...")
    
    config = {
        # Memory system settings
        'enable_context_memory': True,
        'enable_audio_metadata': True,
        'conversation_db_path': 'example_voice_conversations.db',
        'conversation_memory_limit': 50,
        'audio_history_limit': 100,
        
        # Manager features
        'enable_memory_fusion': True,
        'enable_pattern_learning': True,
        'enable_async_processing': True,
        'performance_tracking': True,
        
        # Learning and analysis
        'pattern_learning_threshold': 2,
        'confidence_threshold': 0.7,
        'wake_word_learning_enabled': True,
        
        # Performance
        'max_context_window': 8,
        'auto_save_interval': 300  # 5 minutes
    }
    
    manager = create_voice_memory_manager(config)
    print(f"✅ Manager initialized with session: {manager.current_session_id}")
    
    # 2. Simulate voice interactions
    print("\n2. Adding voice interactions...")
    
    # Example 1: Weather query
    weather_interaction = simulate_voice_interaction(
        manager,
        user_input="What's the weather like today?",
        assistant_response="The current weather is 72°F and sunny with light winds.",
        command_type="question",
        agent_used="weather_agent",
        confidence=0.92,
        wake_word_used=True
    )
    print(f"   Weather query: {weather_interaction['success']}")
    
    # Example 2: Time query
    time_interaction = simulate_voice_interaction(
        manager,
        user_input="What time is it?",
        assistant_response="The current time is 3:45 PM.",
        command_type="question", 
        agent_used="time_agent",
        confidence=0.89,
        wake_word_used=True
    )
    print(f"   Time query: {time_interaction['success']}")
    
    # Example 3: Control command
    control_interaction = simulate_voice_interaction(
        manager,
        user_input="Turn on the lights in the living room",
        assistant_response="I'll turn on the living room lights for you.",
        command_type="command",
        agent_used="home_control_agent",
        confidence=0.85,
        wake_word_used=True
    )
    print(f"   Control command: {control_interaction['success']}")
    
    # Example 4: Information request
    info_interaction = simulate_voice_interaction(
        manager,
        user_input="Tell me about artificial intelligence",
        assistant_response="Artificial intelligence is a branch of computer science that aims to create intelligent machines...",
        command_type="request",
        agent_used="knowledge_agent",
        confidence=0.94,
        wake_word_used=False  # This one wasn't wake word triggered
    )
    print(f"   Information request: {info_interaction['success']}")
    
    # 3. Demonstrate memory fusion
    print("\n3. Memory fusion and context retrieval...")
    
    fused_context = manager.get_fused_context(
        context_window_size=5,
        include_audio_metadata=True,
        include_visual_context=True
    )
    
    print(f"   Conversation history: {len(fused_context['conversation_history'])} interactions")
    print(f"   Memory fusion calls: {fused_context['memory_stats']['memory_fusion_calls']}")
    
    # Get contextual hints for a new query
    hints = manager.get_contextual_response_hints("What's the temperature outside?")
    print(f"   Similar past interactions: {len(hints['similar_past_interactions'])}")
    
    # 4. Search and retrieval
    print("\n4. Search and retrieval...")
    
    # Search for weather-related interactions
    weather_search = manager.search_voice_interactions(
        query="weather",
        limit=5,
        filters={'min_confidence': 0.8}
    )
    print(f"   Weather search results: {weather_search['total_results']}")
    
    # Get recent interactions
    recent = manager.get_recent_interactions(
        limit=3,
        voice_only=True,
        include_metadata=True
    )
    print(f"   Recent voice interactions: {len(recent)}")
    
    # 5. Analytics and insights
    print("\n5. Analytics and insights...")
    
    analytics = manager.get_comprehensive_analytics(days=1)
    
    # System performance
    system_perf = analytics.get('system_performance', {})
    print(f"   Total interactions: {system_perf.get('total_interactions', 0)}")
    print(f"   Successful interactions: {system_perf.get('successful_interactions', 0)}")
    print(f"   Average response time: {system_perf.get('average_response_time', 0):.2f}s")
    
    # Voice analytics
    voice_summary = analytics.get('voice_interaction_summary', {})
    if voice_summary:
        print(f"   Average confidence: {voice_summary.get('average_confidence', 0):.3f}")
        print(f"   Wake word rate: {voice_summary.get('wake_word_rate', 0):.3f}")
    
    # Recommendations
    recommendations = analytics.get('recommendations', [])
    print(f"   Recommendations: {len(recommendations)}")
    for i, rec in enumerate(recommendations[:2], 1):
        print(f"     {i}. {rec['message']}")
    
    # 6. Pattern learning demonstration
    print("\n6. Pattern learning insights...")
    
    pattern_insights = manager._get_pattern_insights()
    top_patterns = pattern_insights.get('top_command_types', [])
    
    if top_patterns:
        print("   Top command patterns:")
        for i, pattern in enumerate(top_patterns[:3], 1):
            print(f"     {i}. {pattern['intent']} - {pattern['frequency']} times")
    else:
        print("   Pattern learning in progress...")
    
    # 7. Session export
    print("\n7. Session export...")
    
    export_data = manager.export_session_data(
        format="json",
        include_analytics=True
    )
    
    conversations = export_data.get('conversation_data', {}).get('conversations', [])
    print(f"   Exported {len(conversations)} conversations")
    
    # Also export as text for human readability
    text_export = manager.export_session_data(format="text")
    print(f"   Text export length: {len(text_export)} characters")
    
    # 8. Memory management
    print("\n8. Memory management...")
    
    # Save all memory
    save_results = manager.save_all_memory()
    print(f"   Memory saved - Context: {save_results['context_memory']}, Conversation: {save_results['conversation_memory']}")
    
    # Get memory stats
    memory_stats = manager._get_memory_fusion_stats()
    print(f"   Context memory available: {memory_stats['context_memory_available']}")
    print(f"   Conversation memory available: {memory_stats['conversation_memory_available']}")
    
    # 9. Cleanup
    print("\n9. Cleanup...")
    manager.close()
    print("   Manager closed successfully")
    
    print("\n✅ Voice Memory Manager example completed successfully!")
    print("\nKey Benefits Demonstrated:")
    print("  • Unified interface for multiple memory systems")
    print("  • Automatic pattern learning and analysis")
    print("  • Comprehensive analytics and insights")
    print("  • Memory fusion for rich context")
    print("  • Flexible search and retrieval")
    print("  • Session management and export")


def simulate_voice_interaction(manager, user_input, assistant_response, 
                             command_type, agent_used, confidence, wake_word_used):
    """
    Simulate a voice interaction with realistic metadata
    
    Args:
        manager: VoiceMemoryManager instance
        user_input: User's spoken input (transcribed)
        assistant_response: Assistant's response
        command_type: Type of command
        agent_used: Which agent processed the request
        confidence: Transcription confidence score
        wake_word_used: Whether wake word was used
        
    Returns:
        Result dictionary from memory manager
    """
    
    # Simulate audio metadata
    audio_metadata = {
        'audio_file_path': f'/tmp/audio_{int(time.time())}.wav',
        'confidence_score': confidence,
        'audio_duration': len(user_input) * 0.08,  # Rough estimate based on text length
        'processing_time': 1.2 + (len(user_input) * 0.01),  # Longer text = more processing
        'wake_word_detected': wake_word_used,
        'wake_word_confidence': 0.87 if wake_word_used else None,
        'language': 'en-US',
        'segments': [
            {
                'start': 0.0,
                'end': len(user_input) * 0.08,
                'text': user_input
            }
        ]
    }
    
    # Simulate processing times
    processing_times = {
        'wake_word_time': 0.1 if wake_word_used else 0.0,
        'transcription_time': 0.6 + (len(user_input) * 0.005),
        'llm_time': 0.4 + (len(assistant_response) * 0.003),
        'tts_time': 0.3 + (len(assistant_response) * 0.002),
        'total_time': 1.4 + (len(user_input) * 0.01) + (len(assistant_response) * 0.005)
    }
    
    # Simulate audio quality metrics
    audio_quality = {
        'noise_level': 0.15 + (0.1 * (1 - confidence)),  # Lower confidence = more noise
        'snr': 15 + (confidence * 10),  # Higher confidence = better SNR
        'clarity': confidence * 0.9,  # Clarity correlated with confidence
        'noise_type': 'ambient',
        'voice_activity_ratio': 0.75
    }
    
    # Add the interaction to memory
    result = manager.add_voice_interaction(
        user_input=user_input,
        assistant_response=assistant_response,
        audio_metadata=audio_metadata,
        agent_used=agent_used,
        command_type=command_type,
        success=True,
        processing_times=processing_times,
        audio_quality=audio_quality
    )
    
    # Add wake word event if used
    if wake_word_used:
        manager.add_wake_word_event(
            wake_word="hey assistant",
            confidence=audio_metadata['wake_word_confidence'],
            conversation_id=result.get('conversation_id'),
            detection_context={
                'detection_latency': 0.12,
                'background_context': 'quiet_room'
            }
        )
    
    return result


if __name__ == "__main__":
    main()