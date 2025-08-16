#!/usr/bin/env python3
"""
Voice Memory Integration Example
Demonstrates how to integrate the voice conversation memory system
with the whisper transcription pipeline and AI providers.
"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import voice memory system
from voice_conversation_memory import VoiceConversationMemory, create_voice_memory

# Import existing whisper system components (mock imports for example)
# In real implementation, these would import from the actual modules
class MockWhisperTranscriber:
    """Mock whisper transcriber for example purposes"""
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        return {
            "text": "What's the weather like today?",
            "confidence": 0.95,
            "language": "en",
            "duration": 2.3,
            "segments": []
        }

class MockAIProvider:
    """Mock AI provider for example purposes"""
    def get_response(self, prompt: str, context: str = None) -> str:
        return f"Based on your question about weather, here's a response..."

class MockWakeWordDetector:
    """Mock wake word detector for example purposes"""
    def detect_wake_word(self, audio_path: str) -> Dict[str, Any]:
        return {
            "detected": True,
            "confidence": 0.87,
            "wake_word": "hey assistant",
            "timestamp": 1.2,
            "detection_time": 0.15
        }


class VoiceMemoryIntegration:
    """
    Integration layer that combines voice conversation memory
    with whisper transcription and AI provider systems.
    """
    
    def __init__(self, 
                 memory_db_path: str = "data/voice_conversations.db",
                 enable_wake_word: bool = True,
                 enable_context: bool = True):
        """
        Initialize voice memory integration
        
        Args:
            memory_db_path: Path to voice conversation database
            enable_wake_word: Enable wake word detection
            enable_context: Enable conversation context
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize voice memory
        self.memory = create_voice_memory(memory_db_path)
        
        # Initialize components (in real implementation, these would be actual classes)
        self.transcriber = MockWhisperTranscriber()
        self.ai_provider = MockAIProvider()
        self.wake_word_detector = MockWakeWordDetector() if enable_wake_word else None
        
        # Configuration
        self.enable_context = enable_context
        self.context_window_size = 5
        
        self.logger.info("✅ Voice memory integration initialized")
    
    def process_voice_input(self, 
                           audio_file_path: str,
                           user_metadata: Dict = None) -> Dict[str, Any]:
        """
        Process a voice input through the complete pipeline with memory integration
        
        Args:
            audio_file_path: Path to audio file
            user_metadata: Additional user metadata
            
        Returns:
            Complete processing result with memory integration
        """
        start_time = time.time()
        result = {
            "success": False,
            "conversation_id": None,
            "processing_times": {},
            "confidence_scores": {},
            "errors": []
        }
        
        try:
            # Step 1: Wake word detection (if enabled)
            wake_word_result = None
            wake_word_time = 0
            
            if self.wake_word_detector:
                wake_start = time.time()
                wake_word_result = self.wake_word_detector.detect_wake_word(audio_file_path)
                wake_word_time = time.time() - wake_start
                
                # Record wake word event
                if wake_word_result.get("detected"):
                    self.memory.add_wake_word_event(
                        wake_word=wake_word_result.get("wake_word", "unknown"),
                        confidence=wake_word_result.get("confidence", 0.0),
                        detection_latency=wake_word_result.get("detection_time", 0.0),
                        background_context=user_metadata.get("environment") if user_metadata else None
                    )
                
                result["wake_word_detected"] = wake_word_result.get("detected", False)
                result["wake_word_confidence"] = wake_word_result.get("confidence", 0.0)
            
            # Step 2: Speech transcription
            transcription_start = time.time()
            transcription_result = self.transcriber.transcribe_audio(audio_file_path)
            transcription_time = time.time() - transcription_start
            
            user_input = transcription_result.get("text", "")
            transcription_confidence = transcription_result.get("confidence", 0.0)
            
            if not user_input:
                raise ValueError("No speech detected in audio")
            
            result["user_input"] = user_input
            result["transcription_confidence"] = transcription_confidence
            
            # Step 3: Get conversation context (if enabled)
            context = ""
            if self.enable_context:
                context = self.memory.get_context_window(
                    size=self.context_window_size,
                    include_metadata=True
                )
            
            # Step 4: AI processing with context
            ai_start = time.time()
            
            # Prepare prompt with context
            prompt = user_input
            if context:
                prompt = f"Context:\n{context}\n\nUser: {user_input}"
            
            ai_response = self.ai_provider.get_response(prompt, context)
            ai_processing_time = time.time() - ai_start
            
            result["assistant_response"] = ai_response
            
            # Step 5: Calculate audio metrics
            audio_duration = transcription_result.get("duration", 0.0)
            audio_analysis = self._analyze_audio_quality(audio_file_path, transcription_result)
            
            # Step 6: Store conversation in memory
            total_processing_time = time.time() - start_time
            
            conversation_id = self.memory.add_voice_conversation(
                user_input=user_input,
                assistant_response=ai_response,
                agent_used="whisper_ai_integration",
                command_type="voice_query",
                success=True,
                response_time=total_processing_time,
                metadata={
                    "user_metadata": user_metadata,
                    "transcription_segments": transcription_result.get("segments", []),
                    "audio_analysis": audio_analysis
                },
                # Voice-specific data
                audio_file_path=audio_file_path,
                audio_duration=audio_duration,
                transcription_confidence=transcription_confidence,
                wake_word_detected=wake_word_result.get("detected", False) if wake_word_result else False,
                wake_word_confidence=wake_word_result.get("confidence") if wake_word_result else None,
                noise_level=audio_analysis.get("noise_level", 0.0),
                voice_activity_ratio=audio_analysis.get("voice_activity_ratio", 0.0),
                # Processing times
                wake_word_detection_time=wake_word_time,
                transcription_time=transcription_time,
                llm_processing_time=ai_processing_time,
                total_processing_time=total_processing_time,
                # Audio quality
                signal_to_noise_ratio=audio_analysis.get("snr"),
                audio_clarity_score=audio_analysis.get("clarity_score"),
                background_noise_type=audio_analysis.get("noise_type"),
                speaking_rate=audio_analysis.get("speaking_rate")
            )
            
            # Step 7: Add detailed transcription quality data
            word_count = len(user_input.split())
            uncertain_words = self._count_uncertain_words(transcription_result)
            
            self.memory.add_transcription_quality(
                conversation_id=conversation_id,
                confidence_score=transcription_confidence,
                word_count=word_count,
                uncertain_words=uncertain_words,
                language_detected=transcription_result.get("language", "unknown")
            )
            
            # Step 8: Learn voice patterns
            self.memory.learn_voice_pattern(
                pattern_type=self._classify_user_input(user_input),
                pattern_text=user_input,
                user_intent=self._extract_intent(user_input),
                context_tags=self._extract_context_tags(user_input, user_metadata)
            )
            
            # Step 9: Register audio file (skip if file doesn't exist in demo)
            if Path(audio_file_path).exists():
                self.memory.add_audio_file(
                    file_path=audio_file_path,
                    conversation_id=conversation_id,
                    audio_analysis=audio_analysis
                )
            
            # Prepare final result
            result.update({
                "success": True,
                "conversation_id": conversation_id,
                "processing_times": {
                    "wake_word_detection": wake_word_time,
                    "transcription": transcription_time,
                    "ai_processing": ai_processing_time,
                    "total": total_processing_time
                },
                "confidence_scores": {
                    "transcription": transcription_confidence,
                    "wake_word": wake_word_result.get("confidence") if wake_word_result else None
                },
                "audio_metrics": audio_analysis
            })
            
            self.logger.info(f"Successfully processed voice input: '{user_input[:50]}...'")
            
        except Exception as e:
            self.logger.error(f"Error processing voice input: {e}")
            result["errors"].append(str(e))
            
            # Still record failed attempt for learning
            error_conv_id = self.memory.add_voice_conversation(
                user_input=result.get("user_input", ""),
                assistant_response=f"Error: {str(e)}",
                agent_used="whisper_ai_integration",
                command_type="voice_query",
                success=False,
                response_time=time.time() - start_time,
                metadata={"error": str(e), "user_metadata": user_metadata},
                audio_file_path=audio_file_path,
                transcription_confidence=result.get("transcription_confidence", 0.0)
            )
            result["conversation_id"] = error_conv_id
        
        return result
    
    def _analyze_audio_quality(self, audio_path: str, transcription_result: Dict) -> Dict[str, Any]:
        """
        Analyze audio quality metrics
        
        Args:
            audio_path: Path to audio file
            transcription_result: Transcription results
            
        Returns:
            Audio quality analysis
        """
        # In real implementation, this would use actual audio analysis libraries
        # like librosa, scipy, or custom audio processing
        
        return {
            "noise_level": 0.2,  # Mock value
            "voice_activity_ratio": 0.75,  # Mock value
            "snr": 15.3,  # Signal-to-noise ratio
            "clarity_score": 0.85,  # Audio clarity score
            "noise_type": "background_chatter",  # Detected noise type
            "speaking_rate": 3.2,  # Words per second
            "pitch_range": "normal",  # Voice pitch characteristics
            "volume_consistency": 0.9,  # Volume level consistency
            "audio_format": "wav",
            "sample_rate": 16000,
            "channels": 1
        }
    
    def _count_uncertain_words(self, transcription_result: Dict) -> int:
        """Count words with low confidence in transcription"""
        # In real implementation, analyze per-word confidence scores
        return 1  # Mock value
    
    def _classify_user_input(self, user_input: str) -> str:
        """Classify the type of user input"""
        user_input_lower = user_input.lower()
        
        if "?" in user_input or any(word in user_input_lower for word in ["what", "how", "when", "where", "why", "who"]):
            return "question"
        elif any(word in user_input_lower for word in ["turn", "set", "start", "stop", "play", "pause"]):
            return "command"
        elif any(word in user_input_lower for word in ["hello", "hi", "hey", "thanks", "thank you"]):
            return "greeting"
        else:
            return "general"
    
    def _extract_intent(self, user_input: str) -> str:
        """Extract user intent from input"""
        user_input_lower = user_input.lower()
        
        # Simple intent extraction (in real implementation, use NLP models)
        if "weather" in user_input_lower:
            return "weather_query"
        elif "time" in user_input_lower:
            return "time_query"
        elif "music" in user_input_lower or "song" in user_input_lower:
            return "music_control"
        elif "light" in user_input_lower:
            return "smart_home_lighting"
        else:
            return "general_query"
    
    def _extract_context_tags(self, user_input: str, user_metadata: Dict = None) -> List[str]:
        """Extract context tags for pattern learning"""
        tags = []
        
        # Add tags based on input content
        if "weather" in user_input.lower():
            tags.append("weather")
        if "time" in user_input.lower():
            tags.append("time")
        if "music" in user_input.lower():
            tags.append("music")
        
        # Add tags based on user metadata
        if user_metadata:
            if user_metadata.get("location"):
                tags.append(f"location_{user_metadata['location']}")
            if user_metadata.get("time_of_day"):
                tags.append(f"time_{user_metadata['time_of_day']}")
        
        return tags
    
    def get_conversation_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive analytics for the integration"""
        voice_analytics = self.memory.get_voice_analytics(days)
        quality_report = self.memory.get_transcription_quality_report(days)
        
        return {
            "analytics_period_days": days,
            "voice_analytics": voice_analytics,
            "transcription_quality": quality_report,
            "generated_at": datetime.now().isoformat()
        }
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for optimizing the voice interaction system"""
        suggestions = []
        
        # Get voice pattern optimization suggestions
        pattern_suggestions = self.memory.optimize_voice_patterns()
        suggestions.extend(pattern_suggestions)
        
        # Get analytics for additional suggestions
        analytics = self.memory.get_voice_analytics(days=7)
        basic_stats = analytics.get("basic_stats", {})
        
        # Add suggestions based on analytics
        if basic_stats.get("low_confidence_rate", 0) > 0.2:
            suggestions.append({
                "type": "improve_audio_quality",
                "priority": "high",
                "suggestion": "High rate of low-confidence transcriptions detected. Consider improving microphone setup or reducing background noise."
            })
        
        if basic_stats.get("wake_word_rate", 0) < 0.7:
            suggestions.append({
                "type": "wake_word_tuning",
                "priority": "medium",
                "suggestion": "Low wake word detection rate. Consider adjusting wake word sensitivity or training."
            })
        
        wake_word_perf = analytics.get("wake_word_performance", {})
        if wake_word_perf.get("false_positive_rate", 0) > 0.1:
            suggestions.append({
                "type": "reduce_false_positives",
                "priority": "medium",
                "suggestion": "High false positive rate for wake word detection. Consider increasing detection threshold."
            })
        
        return suggestions
    
    def export_session_data(self, session_id: str = None, format: str = "json") -> Any:
        """Export session data with integration metrics"""
        return self.memory.export_voice_session(session_id, format)
    
    def close(self):
        """Clean up resources"""
        if self.memory:
            self.memory.close()
        self.logger.info("Voice memory integration closed")


def demo_integration():
    """Demonstrate the voice memory integration"""
    print("🎤 Voice Memory Integration Demo")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize integration
    integration = VoiceMemoryIntegration(
        memory_db_path="demo_voice_conversations.db",
        enable_wake_word=True,
        enable_context=True
    )
    
    try:
        # Simulate processing several voice inputs
        test_scenarios = [
            {
                "audio_path": "/tmp/weather_query.wav",
                "metadata": {"environment": "quiet_room", "time_of_day": "morning"}
            },
            {
                "audio_path": "/tmp/music_request.wav", 
                "metadata": {"environment": "background_music", "time_of_day": "evening"}
            },
            {
                "audio_path": "/tmp/time_query.wav",
                "metadata": {"environment": "kitchen", "time_of_day": "afternoon"}
            }
        ]
        
        results = []
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📝 Processing voice input {i}...")
            
            result = integration.process_voice_input(
                audio_file_path=scenario["audio_path"],
                user_metadata=scenario["metadata"]
            )
            
            results.append(result)
            
            if result["success"]:
                print(f"✅ Success! Conversation ID: {result['conversation_id']}")
                print(f"   User: {result['user_input']}")
                print(f"   Assistant: {result['assistant_response'][:100]}...")
                print(f"   Confidence: {result['transcription_confidence']:.3f}")
                print(f"   Processing Time: {result['processing_times']['total']:.3f}s")
            else:
                print(f"❌ Failed: {result['errors']}")
        
        # Show analytics
        print(f"\n📊 Analytics Summary")
        print("-" * 30)
        
        analytics = integration.get_conversation_analytics(days=1)
        basic_stats = analytics["voice_analytics"]["basic_stats"]
        
        print(f"Total Conversations: {basic_stats['total_conversations']}")
        print(f"Average Confidence: {basic_stats['average_confidence']:.3f}")
        print(f"Wake Word Rate: {basic_stats['wake_word_rate']:.3f}")
        print(f"Average Processing Time: {basic_stats['average_processing_time']:.3f}s")
        
        # Show optimization suggestions
        suggestions = integration.get_optimization_suggestions()
        if suggestions:
            print(f"\n💡 Optimization Suggestions")
            print("-" * 30)
            for suggestion in suggestions:
                print(f"• {suggestion['suggestion']}")
        
        # Export session data
        print(f"\n📄 Session Export Example")
        print("-" * 30)
        session_data = integration.export_session_data(format="json")
        print(f"Exported {len(session_data['conversations'])} conversations")
        print(f"Session started: {session_data['session']['start_time']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        integration.close()
        print(f"\n✅ Demo completed!")


if __name__ == "__main__":
    demo_integration()