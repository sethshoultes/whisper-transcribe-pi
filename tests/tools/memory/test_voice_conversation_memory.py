#!/usr/bin/env python3
"""
Test suite for Voice Conversation Memory System
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3

# Add the project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.memory.voice_conversation_memory import VoiceConversationMemory, create_voice_memory


class TestVoiceConversationMemory(unittest.TestCase):
    """Test cases for VoiceConversationMemory class"""
    
    def setUp(self):
        """Set up test database and memory instance"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.test_db.close()
        self.memory = VoiceConversationMemory(self.test_db.name)
        
    def tearDown(self):
        """Clean up test database"""
        self.memory.close()
        os.unlink(self.test_db.name)
    
    def test_initialization(self):
        """Test memory system initialization"""
        self.assertIsNotNone(self.memory.conn)
        self.assertIsNotNone(self.memory.current_session_id)
        self.assertTrue(Path(self.test_db.name).exists())
        
        # Check that all tables were created
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'voice_conversations',
            'voice_sessions', 
            'audio_files',
            'wake_word_events',
            'transcription_quality',
            'voice_patterns',
            'voice_user_preferences',
            'voice_conversations_fts'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} was not created")
    
    def test_add_voice_conversation(self):
        """Test adding voice conversations"""
        conv_id = self.memory.add_voice_conversation(
            user_input="Test user input",
            assistant_response="Test assistant response",
            agent_used="test_agent",
            command_type="voice_command",
            success=True,
            response_time=1.5,
            audio_file_path="/test/audio.wav",
            audio_duration=3.2,
            transcription_confidence=0.95,
            wake_word_detected=True,
            wake_word_confidence=0.87,
            noise_level=0.3,
            voice_activity_ratio=0.8,
            transcription_time=0.5,
            llm_processing_time=1.0,
            total_processing_time=2.1
        )
        
        self.assertIsInstance(conv_id, int)
        self.assertGreater(conv_id, 0)
        
        # Verify conversation was stored
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT * FROM voice_conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row["user_input"], "Test user input")
        self.assertEqual(row["assistant_response"], "Test assistant response")
        self.assertEqual(row["transcription_confidence"], 0.95)
        self.assertEqual(row["wake_word_detected"], 1)
        self.assertEqual(row["audio_duration"], 3.2)
    
    def test_get_recent_conversations(self):
        """Test retrieving recent conversations"""
        # Add multiple conversations
        for i in range(5):
            self.memory.add_voice_conversation(
                user_input=f"User message {i}",
                assistant_response=f"Assistant response {i}",
                transcription_confidence=0.9 + (i * 0.01),
                audio_duration=2.0 + i
            )
        
        recent = self.memory.get_recent(3)
        self.assertEqual(len(recent), 3)
        
        # Check that they're in chronological order (most recent last)
        self.assertEqual(recent[-1]["user"], "User message 4")
        self.assertEqual(recent[-2]["user"], "User message 3")
        self.assertEqual(recent[-3]["user"], "User message 2")
        
        # Test session-only retrieval
        recent_session = self.memory.get_recent(10, session_only=True)
        self.assertEqual(len(recent_session), 5)
    
    def test_voice_search(self):
        """Test voice conversation search functionality"""
        # Add conversations with different content
        self.memory.add_voice_conversation(
            user_input="What's the weather today?",
            assistant_response="It's sunny with 75 degrees",
            transcription_confidence=0.95
        )
        
        self.memory.add_voice_conversation(
            user_input="Play some music",
            assistant_response="Playing your favorite playlist",
            transcription_confidence=0.88
        )
        
        # Search for weather
        results = self.memory.search_voice_conversations("weather")
        self.assertEqual(len(results), 1)
        self.assertIn("weather", results[0]["user"].lower())
        
        # Search with confidence filter
        results_high_conf = self.memory.search_voice_conversations(
            "weather", 
            min_confidence=0.9
        )
        self.assertEqual(len(results_high_conf), 1)
        
        # Search with confidence filter that excludes results
        results_filtered = self.memory.search_voice_conversations(
            "music", 
            min_confidence=0.9
        )
        self.assertEqual(len(results_filtered), 0)
    
    def test_audio_file_management(self):
        """Test audio file tracking"""
        # Create a temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(b"fake audio data for testing")
        temp_audio.close()
        
        try:
            # Test adding audio file
            audio_analysis = {
                "duration": 5.2,
                "sample_rate": 16000,
                "channels": 1,
                "noise_profile": {"background_noise": 0.1},
                "quality": {"clarity_score": 0.85}
            }
            
            audio_id = self.memory.add_audio_file(
                temp_audio.name,
                audio_analysis=audio_analysis
            )
            
            self.assertIsInstance(audio_id, int)
            
            # Verify audio file record
            cursor = self.memory.conn.cursor()
            cursor.execute("SELECT * FROM audio_files WHERE id = ?", (audio_id,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            self.assertEqual(row["duration"], 5.2)
            self.assertEqual(row["sample_rate"], 16000)
            self.assertIsNotNone(row["file_hash"])
            
        finally:
            os.unlink(temp_audio.name)
    
    def test_wake_word_events(self):
        """Test wake word event tracking"""
        event_id = self.memory.add_wake_word_event(
            wake_word="hey assistant",
            confidence=0.92,
            false_positive=False,
            detection_latency=0.3,
            background_context="quiet room"
        )
        
        self.assertIsInstance(event_id, int)
        
        # Verify event was stored
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT * FROM wake_word_events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row["wake_word"], "hey assistant")
        self.assertEqual(row["confidence"], 0.92)
        self.assertEqual(row["false_positive"], 0)
    
    def test_transcription_quality(self):
        """Test transcription quality tracking"""
        # Add a conversation first
        conv_id = self.memory.add_voice_conversation(
            user_input="Test conversation",
            assistant_response="Test response"
        )
        
        # Add quality data
        quality_id = self.memory.add_transcription_quality(
            conversation_id=conv_id,
            confidence_score=0.93,
            word_count=15,
            uncertain_words=2,
            correction_needed=False,
            language_detected="en-US"
        )
        
        self.assertIsInstance(quality_id, int)
        
        # Verify quality record
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT * FROM transcription_quality WHERE id = ?", (quality_id,))
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row["confidence_score"], 0.93)
        self.assertEqual(row["word_count"], 15)
        self.assertEqual(row["language_detected"], "en-US")
    
    def test_voice_pattern_learning(self):
        """Test voice pattern learning and storage"""
        pattern_id = self.memory.learn_voice_pattern(
            pattern_type="command",
            pattern_text="Turn on the lights",
            user_intent="smart_home_control",
            context_tags=["lighting", "smart_home"]
        )
        
        self.assertIsInstance(pattern_id, int)
        
        # Add the same pattern again to test frequency update
        pattern_id2 = self.memory.learn_voice_pattern(
            pattern_type="command",
            pattern_text="Turn on the lights"
        )
        
        self.assertEqual(pattern_id, pattern_id2)
        
        # Verify pattern and frequency
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT * FROM voice_patterns WHERE id = ?", (pattern_id,))
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row["frequency"], 2)
        self.assertEqual(row["pattern_text"], "Turn on the lights")
    
    def test_voice_analytics(self):
        """Test voice analytics generation"""
        # Add multiple conversations with different characteristics
        conversations_data = [
            {
                "user_input": "Weather check",
                "assistant_response": "Sunny today",
                "transcription_confidence": 0.95,
                "wake_word_detected": True,
                "audio_duration": 2.5,
                "noise_level": 0.2,
                "total_processing_time": 1.8
            },
            {
                "user_input": "Play music",
                "assistant_response": "Playing music",
                "transcription_confidence": 0.85,
                "wake_word_detected": False,
                "audio_duration": 1.8,
                "noise_level": 0.4,
                "total_processing_time": 1.2
            },
            {
                "user_input": "Set timer",
                "assistant_response": "Timer set",
                "transcription_confidence": 0.92,
                "wake_word_detected": True,
                "audio_duration": 1.5,
                "noise_level": 0.1,
                "total_processing_time": 0.9
            }
        ]
        
        for conv_data in conversations_data:
            self.memory.add_voice_conversation(**conv_data)
        
        # Get analytics
        analytics = self.memory.get_voice_analytics(days=1)
        
        self.assertIn("basic_stats", analytics)
        self.assertIn("processing_performance", analytics)
        self.assertIn("wake_word_performance", analytics)
        
        basic_stats = analytics["basic_stats"]
        self.assertEqual(basic_stats["total_conversations"], 3)
        self.assertEqual(basic_stats["wake_word_conversations"], 2)
        self.assertAlmostEqual(basic_stats["wake_word_rate"], 0.667, places=2)
        self.assertGreater(basic_stats["average_confidence"], 0.8)
    
    def test_transcription_quality_report(self):
        """Test transcription quality reporting"""
        # Add conversations with quality data
        conv_ids = []
        for i in range(3):
            conv_id = self.memory.add_voice_conversation(
                user_input=f"Test message {i}",
                assistant_response=f"Response {i}"
            )
            conv_ids.append(conv_id)
            
            self.memory.add_transcription_quality(
                conversation_id=conv_id,
                confidence_score=0.9 - (i * 0.05),
                word_count=10 + i,
                uncertain_words=i,
                language_detected="en-US"
            )
        
        report = self.memory.get_transcription_quality_report(days=1)
        
        self.assertIn("quality_metrics", report)
        self.assertIn("summary", report)
        
        summary = report["summary"]
        self.assertEqual(summary["total_transcriptions"], 3)
        self.assertGreater(summary["overall_avg_confidence"], 0.8)
    
    def test_session_export(self):
        """Test session data export"""
        # Add some conversations
        self.memory.add_voice_conversation(
            user_input="Export test",
            assistant_response="Export response",
            transcription_confidence=0.95
        )
        
        # Test JSON export
        json_export = self.memory.export_voice_session(format="json")
        
        self.assertIsInstance(json_export, dict)
        self.assertIn("session", json_export)
        self.assertIn("conversations", json_export)
        self.assertIn("wake_word_events", json_export)
        self.assertIn("exported_at", json_export)
        
        # Test text export
        text_export = self.memory.export_voice_session(format="text")
        
        self.assertIsInstance(text_export, str)
        self.assertIn("Voice Session:", text_export)
        self.assertIn("Export test", text_export)
    
    def test_optimization_suggestions(self):
        """Test voice pattern optimization suggestions"""
        # Add patterns with different success rates
        self.memory.learn_voice_pattern(
            pattern_type="command",
            pattern_text="Problematic command"
        )
        
        # Manually update success rate to simulate poor performance
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            UPDATE voice_patterns 
            SET frequency = 5, success_rate = 0.6 
            WHERE pattern_text = 'Problematic command'
        """)
        
        suggestions = self.memory.optimize_voice_patterns(min_frequency=3)
        
        self.assertIsInstance(suggestions, list)
        if suggestions:  # Only check if we have suggestions
            self.assertEqual(suggestions[0]["type"], "improve_pattern")
            self.assertIn("Problematic command", suggestions[0]["pattern"])
    
    def test_create_voice_memory_function(self):
        """Test the convenience function"""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        try:
            memory = create_voice_memory(temp_db.name)
            self.assertIsInstance(memory, VoiceConversationMemory)
            self.assertTrue(Path(temp_db.name).exists())
            memory.close()
        finally:
            os.unlink(temp_db.name)
    
    def test_memory_persistence(self):
        """Test that data persists between memory instances"""
        # Add data to first instance
        conv_id = self.memory.add_voice_conversation(
            user_input="Persistence test",
            assistant_response="Will this persist?",
            transcription_confidence=0.98
        )
        
        session_id = self.memory.current_session_id
        self.memory.close()
        
        # Create new instance with same database
        memory2 = VoiceConversationMemory(self.test_db.name)
        
        # Check that data persists
        cursor = memory2.conn.cursor()
        cursor.execute("SELECT * FROM voice_conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row["user_input"], "Persistence test")
        
        # Check that sessions persist
        cursor.execute("SELECT * FROM voice_sessions WHERE id = ?", (session_id,))
        session_row = cursor.fetchone()
        self.assertIsNotNone(session_row)
        
        memory2.close()


class TestVoiceMemoryIntegration(unittest.TestCase):
    """Integration tests for voice memory system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.test_db.close()
        self.memory = VoiceConversationMemory(self.test_db.name)
    
    def tearDown(self):
        """Clean up"""
        self.memory.close()
        os.unlink(self.test_db.name)
    
    def test_complete_voice_interaction_flow(self):
        """Test a complete voice interaction from start to finish"""
        # Simulate wake word detection
        wake_event_id = self.memory.add_wake_word_event(
            wake_word="hey assistant",
            confidence=0.95,
            detection_latency=0.2
        )
        
        # Simulate conversation
        conv_id = self.memory.add_voice_conversation(
            user_input="What's the time?",
            assistant_response="It's 3:30 PM",
            transcription_confidence=0.92,
            wake_word_detected=True,
            wake_word_confidence=0.95,
            audio_duration=2.1,
            noise_level=0.25,
            wake_word_detection_time=0.2,
            transcription_time=0.6,
            llm_processing_time=0.8,
            tts_generation_time=0.4,
            total_processing_time=2.0
        )
        
        # Add transcription quality
        quality_id = self.memory.add_transcription_quality(
            conversation_id=conv_id,
            confidence_score=0.92,
            word_count=4,
            uncertain_words=0,
            language_detected="en-US"
        )
        
        # Learn the pattern
        pattern_id = self.memory.learn_voice_pattern(
            pattern_type="question",
            pattern_text="What's the time?",
            user_intent="time_query"
        )
        
        # Verify all components work together
        self.assertIsInstance(wake_event_id, int)
        self.assertIsInstance(conv_id, int)
        self.assertIsInstance(quality_id, int)
        self.assertIsInstance(pattern_id, int)
        
        # Test analytics with complete data
        analytics = self.memory.get_voice_analytics(days=1)
        self.assertEqual(analytics["basic_stats"]["total_conversations"], 1)
        self.assertEqual(analytics["basic_stats"]["wake_word_conversations"], 1)
        
        # Test search
        search_results = self.memory.search_voice_conversations("time")
        self.assertEqual(len(search_results), 1)
        
        # Test export
        export_data = self.memory.export_voice_session(format="json")
        self.assertEqual(len(export_data["conversations"]), 1)
        self.assertEqual(len(export_data["wake_word_events"]), 1)


if __name__ == "__main__":
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run tests
    unittest.main(verbosity=2)