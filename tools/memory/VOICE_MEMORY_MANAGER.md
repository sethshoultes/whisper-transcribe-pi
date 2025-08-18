# Voice Memory Manager

The **VoiceMemoryManager** is the central coordination point for all voice memory operations in the whisper-transcribe-pi project. It provides a unified interface that coordinates between VoiceContextMemory and VoiceConversationMemory systems to deliver comprehensive voice interaction management.

## 🎯 Key Features

### Unified Memory Interface
- **Dual Memory Coordination**: Seamlessly coordinates VoiceContextMemory and VoiceConversationMemory
- **Automatic Synchronization**: Keeps both memory systems in sync during voice interactions
- **Fallback Support**: Continues operation even if one memory system fails

### Memory Fusion
- **Comprehensive Context**: Combines data from both memory systems for rich context
- **Smart Context Windows**: Configurable context window sizes for optimal performance
- **Cross-System Analytics**: Generates insights by analyzing data across both systems

### Pattern Learning & Analysis
- **Automatic Pattern Recognition**: Learns voice command patterns and user preferences
- **Intent Classification**: Automatically classifies command types and intents
- **Success Rate Tracking**: Monitors and improves interaction success rates

### Advanced Analytics
- **Performance Metrics**: Tracks response times, confidence scores, and success rates
- **Audio Quality Analysis**: Monitors transcription quality and audio conditions
- **Usage Patterns**: Identifies peak usage times and common interaction types
- **Recommendation Engine**: Provides suggestions for system improvements

### Session Management
- **Session Tracking**: Maintains session state across voice interactions
- **Data Export**: Export session data in JSON or text formats
- **Memory Cleanup**: Efficient memory management and cleanup operations

## 🚀 Quick Start

### Basic Usage

```python
from tools.memory import create_voice_memory_manager

# Create manager with default configuration
manager = create_voice_memory_manager()

# Add a voice interaction
result = manager.add_voice_interaction(
    user_input="What's the weather like?",
    assistant_response="The weather is sunny and 75°F",
    audio_metadata={
        'confidence_score': 0.92,
        'audio_duration': 2.3,
        'wake_word_detected': True
    },
    agent_used="weather_agent",
    command_type="question"
)

# Get fused context for LLM
context = manager.get_fused_context(context_window_size=5)

# Clean shutdown
manager.close()
```

### Custom Configuration

```python
config = {
    'enable_context_memory': True,
    'enable_audio_metadata': True,
    'conversation_db_path': 'my_voice_conversations.db',
    'enable_pattern_learning': True,
    'enable_async_processing': True,
    'max_context_window': 10,
    'confidence_threshold': 0.7
}

manager = create_voice_memory_manager(config)
```

## 📊 Memory Fusion

The VoiceMemoryManager's memory fusion capability combines data from both memory systems:

```python
# Get comprehensive fused context
fused_context = manager.get_fused_context(
    context_window_size=8,
    include_audio_metadata=True,
    include_visual_context=True,
    voice_only=False
)

# Contains:
# - conversation_history: Recent interactions from both systems
# - visual_context: Current scene state and recent changes
# - voice_analytics: Comprehensive voice interaction metrics
# - pattern_insights: Learned patterns and recommendations
# - memory_stats: System health and performance data
```

## 🔍 Search & Retrieval

### Voice Interaction Search

```python
# Search across both memory systems
results = manager.search_voice_interactions(
    query="weather temperature",
    limit=10,
    search_scope="both",  # "both", "context", or "conversation"
    filters={
        'min_confidence': 0.8,
        'wake_word_only': True
    }
)
```

### Contextual Hints

```python
# Get hints for better response generation
hints = manager.get_contextual_response_hints("What's the temperature?")

# Returns:
# - similar_past_interactions: Related previous conversations
# - relevant_visual_context: Current scene objects and context
# - suggested_response_patterns: Recommended response approaches
# - user_preferences: Learned user preferences
```

## 📈 Analytics & Insights

### Comprehensive Analytics

```python
analytics = manager.get_comprehensive_analytics(days=7)

# Includes:
# - voice_interaction_summary: Basic interaction statistics
# - transcription_quality: Quality metrics and trends
# - wake_word_performance: Wake word detection analytics
# - conversation_patterns: Usage patterns and trends
# - audio_quality_trends: Audio quality over time
# - system_performance: Manager performance metrics
# - recommendations: Improvement suggestions
```

### Real-time Performance

```python
# Get current performance stats
stats = manager.performance_stats

print(f"Total interactions: {stats['total_interactions']}")
print(f"Success rate: {stats['successful_interactions'] / stats['total_interactions']:.2%}")
print(f"Avg response time: {stats['average_response_time']:.2f}s")
```

## 🧠 Pattern Learning

The VoiceMemoryManager automatically learns from voice interactions:

### Automatic Learning
- **Command Patterns**: Learns common phrasings and command structures
- **User Intents**: Classifies and tracks user intentions
- **Success Patterns**: Identifies what leads to successful interactions
- **Context Associations**: Links commands with visual/situational context

### Pattern Insights
```python
insights = manager._get_pattern_insights()

# Returns:
# - top_command_types: Most frequent command patterns
# - success_patterns: High-success interaction patterns  
# - improvement_suggestions: Recommended optimizations
```

## 💾 Session Management

### Session Tracking
```python
# Session information is automatically managed
session_id = manager.current_session_id

# Export current session
export_data = manager.export_session_data(
    format="json",  # or "text"
    include_analytics=True
)
```

### Memory Persistence
```python
# Save all memory systems
save_results = manager.save_all_memory()

# Clear memory with options
clear_results = manager.clear_memory(
    clear_context=True,
    clear_conversation=True,
    keep_preferences=True
)
```

## ⚙️ Configuration Options

### Core Settings
```python
config = {
    # Memory Systems
    'enable_context_memory': True,
    'enable_audio_metadata': True,
    'conversation_db_path': 'voice_conversations.db',
    'conversation_memory_limit': 100,
    'audio_history_limit': 500,
    'conversation_cache_size': 200,
    
    # Manager Features
    'enable_memory_fusion': True,
    'enable_pattern_learning': True,
    'enable_async_processing': True,
    'auto_save_interval': 300,
    'performance_tracking': True,
    
    # Learning & Analysis
    'pattern_learning_threshold': 3,
    'confidence_threshold': 0.7,
    'wake_word_learning_enabled': True,
    'transcription_quality_tracking': True,
    
    # Performance
    'max_context_window': 10,
    'memory_cleanup_interval': 3600,
    'enable_memory_compression': True,
    
    # Error Handling
    'fallback_to_single_memory': True,
    'retry_failed_operations': True,
    'max_retry_attempts': 3
}
```

## 🔧 Integration Examples

### With Whisper Transcription
```python
# After transcription
transcription_result = whisper_model.transcribe(audio_file)

# Add to memory with full metadata
manager.add_voice_interaction(
    user_input=transcription_result['text'],
    assistant_response=llm_response,
    audio_metadata={
        'audio_file_path': audio_file,
        'confidence_score': transcription_result.get('confidence'),
        'audio_duration': transcription_result.get('duration'),
        'language': transcription_result.get('language'),
        'segments': transcription_result.get('segments')
    },
    processing_times={
        'transcription_time': transcription_time,
        'llm_time': llm_processing_time,
        'total_time': total_time
    }
)
```

### With Wake Word Detection
```python
# When wake word is detected
if wake_word_detected:
    manager.add_wake_word_event(
        wake_word="hey assistant",
        confidence=detection_confidence,
        detection_context={
            'detection_latency': latency,
            'background_context': environment_info
        }
    )
```

### With LLM Context
```python
# Get context for LLM prompt
fused_context = manager.get_fused_context(context_window_size=5)
conversation_history = fused_context['conversation_history']

# Build prompt with memory context
prompt = build_llm_prompt(user_input, conversation_history)
response = llm_model.generate(prompt)
```

## 🚨 Error Handling

The VoiceMemoryManager includes robust error handling:

```python
# Automatic fallback if one memory system fails
try:
    result = manager.add_voice_interaction(...)
    if not result['success']:
        print(f"Errors: {result['errors']}")
except Exception as e:
    # Manager continues with available systems
    logger.error(f"Memory operation failed: {e}")
```

## 📋 Best Practices

### 1. Configuration
- Set appropriate `max_context_window` for your use case
- Enable `async_processing` for better performance
- Configure `confidence_threshold` based on your audio quality

### 2. Memory Management
- Call `manager.close()` for clean shutdown
- Use `save_all_memory()` periodically for data persistence
- Monitor memory usage with analytics

### 3. Performance Optimization
- Use `voice_only=True` filters when appropriate
- Limit search results with reasonable `limit` values
- Enable memory compression for large datasets

### 4. Error Recovery
- Enable `fallback_to_single_memory` for resilience
- Monitor error rates in analytics
- Implement retry logic for critical operations

## 📝 Example Applications

### 1. Smart Home Assistant
```python
# Context-aware home control
fused_context = manager.get_fused_context(include_visual_context=True)
current_room_objects = fused_context['visual_context']['current_scene']

# Use context for better command understanding
if "lights" in user_input and "living room" in current_room_objects:
    # Execute living room light control
```

### 2. Personal AI Assistant
```python
# Learn user preferences over time
analytics = manager.get_comprehensive_analytics(days=30)
user_patterns = analytics['conversation_patterns']

# Adapt responses based on learned patterns
if user_patterns['preferred_response_style'] == 'concise':
    # Generate shorter responses
```

### 3. Voice Analytics Dashboard
```python
# Real-time monitoring
performance = manager.get_comprehensive_analytics(days=1)
wake_word_accuracy = performance['wake_word_performance']['false_positive_rate']
transcription_quality = performance['transcription_quality']['summary']['overall_avg_confidence']

# Display metrics in dashboard
```

## 🛠️ Development & Testing

### Running Examples
```bash
# Run the comprehensive example
python tools/memory/voice_memory_manager_example.py

# Run basic functionality test
python -c "
from tools.memory import create_voice_memory_manager
manager = create_voice_memory_manager()
print('Manager initialized successfully')
manager.close()
"
```

### Custom Extensions
The VoiceMemoryManager is designed for extensibility:

```python
class CustomVoiceMemoryManager(VoiceMemoryManager):
    def custom_analysis(self):
        # Add your custom analytics
        pass
    
    def special_pattern_learning(self):
        # Implement specialized learning
        pass
```

## 📚 Related Components

- **VoiceContextMemory**: Enhanced context memory with voice-specific features
- **VoiceConversationMemory**: SQLite-based persistent conversation storage
- **Whisper Integration**: Audio transcription with memory integration
- **LLM Context Building**: Memory-driven context for language models

## 🔗 Integration Points

The VoiceMemoryManager serves as the central hub connecting:
- Audio transcription systems (Whisper)
- Wake word detection engines
- Language models (LLMs) 
- Text-to-speech systems
- Vision systems (for visual context)
- Home automation systems
- Analytics and monitoring tools

## 📈 Future Enhancements

Planned improvements include:
- Machine learning-based pattern recognition
- Multi-user voice profile support
- Advanced audio quality analysis
- Real-time context adaptation
- Cloud synchronization capabilities
- Voice biometric integration

---

The VoiceMemoryManager provides a powerful, flexible foundation for building sophisticated voice-controlled applications with comprehensive memory management, pattern learning, and analytics capabilities.