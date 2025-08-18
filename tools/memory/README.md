# Voice Conversation Memory System

Production-ready SQLite-based conversation memory system specifically designed for voice interactions in whisper-transcribe-pi. Built from the original ConversationMemory from ai-assistant-project with comprehensive voice-specific enhancements.

## Features

### Core SQLite Database Features
- **Production-ready SQLite backend** with full ACID compliance
- **Multi-threaded access** support with proper connection handling
- **Full-text search** capabilities for conversation content
- **Automatic schema versioning** and migration support
- **Data integrity** with foreign key constraints and indexes

### Voice-Specific Database Schema
- **Voice conversations table** with audio metadata, confidence scores, and processing times
- **Audio files tracking** with file hashes, quality metrics, and analysis results
- **Wake word events** with detection confidence and timing data
- **Transcription quality** monitoring with word-level confidence tracking
- **Voice patterns learning** for command optimization and intent recognition
- **Voice sessions** with comprehensive performance metrics

### Analytics and Insights
- **Real-time analytics** for voice interactions, quality trends, and performance
- **Wake word performance** analysis with false positive detection
- **Transcription quality** reporting with confidence distributions
- **Voice pattern optimization** suggestions based on usage data
- **Session export** capabilities with full audit trails

## Quick Start

### Installation and Setup

```python
from tools.memory.voice_conversation_memory import create_voice_memory

# Create and initialize voice memory
memory = create_voice_memory("data/voice_conversations.db")
```

### Basic Usage

```python
# Add a voice conversation with detailed metadata
conversation_id = memory.add_voice_conversation(
    user_input="What's the weather like?",
    assistant_response="It's sunny with 75 degrees",
    transcription_confidence=0.95,
    wake_word_detected=True,
    wake_word_confidence=0.87,
    audio_duration=2.3,
    noise_level=0.2,
    transcription_time=0.5,
    llm_processing_time=1.2,
    total_processing_time=1.8
)

# Add detailed transcription quality data
memory.add_transcription_quality(
    conversation_id=conversation_id,
    confidence_score=0.95,
    word_count=6,
    uncertain_words=0,
    language_detected="en-US"
)

# Record wake word event
memory.add_wake_word_event(
    wake_word="hey assistant",
    confidence=0.87,
    detection_latency=0.15
)

# Learn voice patterns for optimization
memory.learn_voice_pattern(
    pattern_type="question",
    pattern_text="What's the weather like?",
    user_intent="weather_query"
)
```

### Advanced Analytics

```python
# Get comprehensive voice analytics
analytics = memory.get_voice_analytics(days=7)
print(f"Total conversations: {analytics['basic_stats']['total_conversations']}")
print(f"Average confidence: {analytics['basic_stats']['average_confidence']}")
print(f"Wake word rate: {analytics['basic_stats']['wake_word_rate']}")

# Get transcription quality report
quality_report = memory.get_transcription_quality_report(days=7)
print(f"Correction rate: {quality_report['summary']['correction_rate']}")

# Search conversations with voice filters
results = memory.search_voice_conversations(
    "weather", 
    min_confidence=0.9,
    wake_word_only=True
)

# Get optimization suggestions
suggestions = memory.optimize_voice_patterns()
for suggestion in suggestions:
    print(f"• {suggestion['suggestion']}")
```

## Database Schema

### Core Tables
- **`voice_conversations`** - Main conversation data with voice metrics
- **`voice_sessions`** - Session tracking with performance analytics
- **`audio_files`** - Audio file metadata and analysis results
- **`wake_word_events`** - Wake word detection events
- **`transcription_quality`** - Transcription quality metrics
- **`voice_patterns`** - Learned voice command patterns
- **`voice_user_preferences`** - User preferences and settings

### Voice-Specific Fields
- Audio duration, file paths, and quality metrics
- Transcription confidence scores and processing times
- Wake word detection confidence and timing
- Background noise levels and voice activity ratios
- Processing time breakdowns (wake word, transcription, LLM, TTS)
- Audio quality metrics (SNR, clarity, noise type)

## Database Management

### Setup and Migration

```bash
# Initialize fresh database
python tools/memory/voice_memory_setup.py init

# Migrate existing database
python tools/memory/voice_memory_setup.py migrate

# Validate database integrity
python tools/memory/voice_memory_setup.py validate

# Get database information
python tools/memory/voice_memory_setup.py info

# Clean up old data
python tools/memory/voice_memory_setup.py cleanup --days 30
```

### Integration Example

```python
from tools.memory.voice_memory_integration_example import VoiceMemoryIntegration

# Initialize complete integration
integration = VoiceMemoryIntegration(
    memory_db_path="data/voice_conversations.db",
    enable_wake_word=True,
    enable_context=True
)

# Process complete voice pipeline
result = integration.process_voice_input(
    audio_file_path="/path/to/audio.wav",
    user_metadata={"environment": "quiet_room"}
)

# Get comprehensive analytics
analytics = integration.get_conversation_analytics(days=7)

# Get optimization suggestions
suggestions = integration.get_optimization_suggestions()
```

## Key Methods

### Core Voice Memory Methods
- **`add_voice_conversation()`** - Add conversation with voice metadata
- **`add_audio_file()`** - Register audio file with analysis
- **`add_wake_word_event()`** - Record wake word detection
- **`add_transcription_quality()`** - Track transcription quality
- **`learn_voice_pattern()`** - Learn voice command patterns

### Analytics and Search
- **`get_voice_analytics()`** - Comprehensive voice interaction analytics
- **`get_transcription_quality_report()`** - Quality metrics and trends
- **`search_voice_conversations()`** - Advanced conversation search
- **`optimize_voice_patterns()`** - Get optimization suggestions

### Data Management
- **`export_voice_session()`** - Export session data
- **`get_recent()`** - Get recent conversations with voice data
- **`get_context_window()`** - Format context for LLM

## Testing

```bash
# Run comprehensive test suite
python -m pytest tests/tools/memory/test_voice_conversation_memory.py -v

# Run integration demo
python tools/memory/voice_memory_integration_example.py
```

## Performance Features

- **SQLite optimization** with proper indexes and query optimization
- **Connection pooling** with thread-safe access patterns
- **Full-text search** using SQLite FTS5 for fast conversation search
- **Automatic cleanup** with configurable data retention policies
- **Schema versioning** for seamless upgrades and migrations
- **Backup and export** capabilities for data portability

## Security and Data Protection

- **No external dependencies** - Pure Python with SQLite
- **Local data storage** - All data remains on device
- **ACID compliance** - Guaranteed data consistency
- **Error handling** - Comprehensive error recovery
- **Data validation** - Input sanitization and type checking