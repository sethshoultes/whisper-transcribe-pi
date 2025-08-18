#!/usr/bin/env python3
"""
Voice Context Memory System - Enhanced context memory for voice interactions
Tracks voice conversations, audio metadata, wake word patterns, and transcription data
Based on the original ContextMemory from ai-assistant-project with voice-specific enhancements
"""

import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import statistics
from collections import defaultdict, Counter

class VoiceContextMemory:
    """Enhanced context memory system specifically designed for voice interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_limit = config.get('conversation_memory_limit', 50)
        self.audio_history_limit = config.get('audio_history_limit', 200)
        self.enable_memory = config.get('enable_context_memory', True)
        self.enable_audio_metadata = config.get('enable_audio_metadata', True)
        
        # Memory storage - inherited from original
        self.conversation_history: List[Dict] = []
        self.visual_context_history: List[Dict] = []
        self.scene_state = {}
        self.user_preferences = {}
        
        # Voice-specific memory storage
        self.audio_metadata_history: List[Dict] = []
        self.wake_word_stats: Dict[str, Any] = {}
        self.voice_command_patterns: Dict[str, Any] = {}
        self.transcription_quality_stats: Dict[str, Any] = {}
        self.voice_interaction_sessions: List[Dict] = []
        
        # File paths for persistence
        self.memory_dir = Path("/home/sethshoultes/whisper-transcribe-pi/data/memory")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Original files
        self.conversation_file = self.memory_dir / "conversations.json"
        self.visual_file = self.memory_dir / "visual_context.json"
        self.preferences_file = self.memory_dir / "user_preferences.json"
        
        # Voice-specific files
        self.audio_metadata_file = self.memory_dir / "audio_metadata.json"
        self.wake_word_file = self.memory_dir / "wake_word_stats.json"
        self.voice_patterns_file = self.memory_dir / "voice_command_patterns.json"
        self.transcription_stats_file = self.memory_dir / "transcription_quality.json"
        self.voice_sessions_file = self.memory_dir / "voice_sessions.json"
        
        # Load existing memory
        self._load_persistent_memory()
        
        print("🧠🎤 Voice Context Memory system initialized")
    
    # ========== Original ContextMemory Methods (Enhanced) ==========
    
    def add_conversation_turn(self, user_input: str, assistant_response: str, 
                            audio_metadata: Optional[Dict] = None):
        """Add a conversation turn to memory with optional audio metadata"""
        if not self.enable_memory:
            return
        
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
            "context_objects": list(self.scene_state.get('current_objects', [])),
            "interaction_type": "voice" if audio_metadata else "text"
        }
        
        # Add audio metadata if provided
        if audio_metadata and self.enable_audio_metadata:
            turn["audio_metadata"] = audio_metadata
            
        self.conversation_history.append(turn)
        
        # Limit memory size
        if len(self.conversation_history) > self.conversation_limit:
            self.conversation_history = self.conversation_history[-self.conversation_limit:]
        
        # Auto-save periodically
        if len(self.conversation_history) % 5 == 0:
            self._save_conversations()
    
    def update_visual_context(self, vision_result: Dict[str, Any]):
        """Update visual context with new detection results (inherited from original)"""
        timestamp = datetime.now()
        current_objects = set(vision_result.get('objects', []))
        
        # Check for scene changes
        previous_objects = set(self.scene_state.get('current_objects', []))
        scene_changed = current_objects != previous_objects
        
        # Update current scene state
        self.scene_state = {
            'current_objects': list(current_objects),
            'last_updated': timestamp.isoformat(),
            'confidence_scores': vision_result.get('confidence_scores', {}),
            'detection_count': vision_result.get('count', len(current_objects))
        }
        
        # Log significant changes
        if scene_changed and (current_objects or previous_objects):
            change_record = {
                'timestamp': timestamp.isoformat(),
                'previous_objects': list(previous_objects),
                'current_objects': list(current_objects),
                'change_type': self._classify_scene_change(previous_objects, current_objects)
            }
            
            self.visual_context_history.append(change_record)
            
            # Limit visual history size
            if len(self.visual_context_history) > 100:
                self.visual_context_history = self.visual_context_history[-100:]
    
    def _classify_scene_change(self, old_objects: set, new_objects: set) -> str:
        """Classify the type of scene change (inherited from original)"""
        if not old_objects and new_objects:
            return "scene_appeared"
        elif old_objects and not new_objects:
            return "scene_cleared"
        elif len(new_objects) > len(old_objects):
            return "objects_added"
        elif len(new_objects) < len(old_objects):
            return "objects_removed"
        else:
            return "objects_changed"
    
    # ========== Voice-Specific Enhancement Methods ==========
    
    def add_audio_metadata(self, audio_file_path: str, transcription_result: Dict[str, Any], 
                          confidence_score: Optional[float] = None, 
                          processing_time: Optional[float] = None):
        """Add audio metadata from transcription process"""
        if not self.enable_audio_metadata:
            return
            
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "audio_file_path": audio_file_path,
            "file_size_bytes": self._get_file_size(audio_file_path),
            "transcription_text": transcription_result.get('text', ''),
            "confidence_score": confidence_score,
            "processing_time_seconds": processing_time,
            "word_count": len(transcription_result.get('text', '').split()),
            "character_count": len(transcription_result.get('text', '')),
            "language_detected": transcription_result.get('language', 'unknown'),
            "segments": transcription_result.get('segments', [])
        }
        
        # Calculate audio duration if available
        if 'segments' in transcription_result and transcription_result['segments']:
            last_segment = transcription_result['segments'][-1]
            metadata["audio_duration_seconds"] = last_segment.get('end', 0)
        
        self.audio_metadata_history.append(metadata)
        
        # Limit audio history size
        if len(self.audio_metadata_history) > self.audio_history_limit:
            self.audio_metadata_history = self.audio_metadata_history[-self.audio_history_limit:]
        
        # Update transcription quality stats
        self._update_transcription_quality_stats(metadata)
        
        # Auto-save periodically
        if len(self.audio_metadata_history) % 10 == 0:
            self._save_audio_metadata()
    
    def track_wake_word_detection(self, wake_word: str, confidence: float, 
                                audio_file_path: Optional[str] = None):
        """Track wake word detection statistics"""
        timestamp = datetime.now()
        
        # Initialize wake word stats if not exists
        if wake_word not in self.wake_word_stats:
            self.wake_word_stats[wake_word] = {
                "total_detections": 0,
                "confidence_scores": [],
                "hourly_distribution": defaultdict(int),
                "daily_distribution": defaultdict(int),
                "first_detected": timestamp.isoformat(),
                "last_detected": timestamp.isoformat()
            }
        
        stats = self.wake_word_stats[wake_word]
        stats["total_detections"] += 1
        stats["confidence_scores"].append(confidence)
        stats["last_detected"] = timestamp.isoformat()
        
        # Track temporal patterns
        hour_key = timestamp.strftime("%H")
        day_key = timestamp.strftime("%A")
        stats["hourly_distribution"][hour_key] += 1
        stats["daily_distribution"][day_key] += 1
        
        # Limit confidence score history
        if len(stats["confidence_scores"]) > 100:
            stats["confidence_scores"] = stats["confidence_scores"][-100:]
        
        # Save wake word stats
        self._save_wake_word_stats()
    
    def analyze_voice_command_patterns(self, command_text: str, intent: Optional[str] = None,
                                     confidence: Optional[float] = None):
        """Analyze and track voice command patterns"""
        timestamp = datetime.now()
        
        # Extract command features
        word_count = len(command_text.split())
        char_count = len(command_text)
        
        # Simple intent classification if not provided
        if not intent:
            intent = self._classify_command_intent(command_text)
        
        # Track command pattern
        pattern_key = f"{intent}_{word_count}_words"
        
        if pattern_key not in self.voice_command_patterns:
            self.voice_command_patterns[pattern_key] = {
                "intent": intent,
                "average_word_count": word_count,
                "examples": [],
                "frequency": 0,
                "confidence_scores": [],
                "first_seen": timestamp.isoformat(),
                "last_seen": timestamp.isoformat()
            }
        
        pattern = self.voice_command_patterns[pattern_key]
        pattern["frequency"] += 1
        pattern["last_seen"] = timestamp.isoformat()
        
        # Add example if not too many stored
        if len(pattern["examples"]) < 5:
            pattern["examples"].append(command_text[:100])  # Truncate long commands
        
        # Track confidence if provided
        if confidence is not None:
            pattern["confidence_scores"].append(confidence)
            if len(pattern["confidence_scores"]) > 50:
                pattern["confidence_scores"] = pattern["confidence_scores"][-50:]
        
        # Update average word count
        pattern["average_word_count"] = statistics.mean([
            len(ex.split()) for ex in pattern["examples"]
        ])
        
        # Save voice patterns
        self._save_voice_patterns()
    
    def start_voice_session(self, session_type: str = "interactive") -> str:
        """Start a new voice interaction session"""
        session_id = f"voice_session_{int(time.time())}"
        
        session = {
            "session_id": session_id,
            "session_type": session_type,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_interactions": 0,
            "total_audio_duration": 0.0,
            "wake_word_detections": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "average_confidence": 0.0,
            "commands_processed": [],
            "context_objects_mentioned": set()
        }
        
        self.voice_interaction_sessions.append(session)
        return session_id
    
    def end_voice_session(self, session_id: str):
        """End a voice interaction session and calculate statistics"""
        for session in self.voice_interaction_sessions:
            if session["session_id"] == session_id:
                session["end_time"] = datetime.now().isoformat()
                
                # Calculate session duration
                start_time = datetime.fromisoformat(session["start_time"])
                end_time = datetime.fromisoformat(session["end_time"])
                session["duration_seconds"] = (end_time - start_time).total_seconds()
                
                # Convert set to list for JSON serialization
                session["context_objects_mentioned"] = list(session["context_objects_mentioned"])
                
                self._save_voice_sessions()
                break
    
    def get_voice_interaction_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of voice interactions in the specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent audio metadata
        recent_audio = [
            audio for audio in self.audio_metadata_history
            if datetime.fromisoformat(audio["timestamp"]) > cutoff_time
        ]
        
        # Filter recent voice conversations
        recent_voice_conversations = [
            conv for conv in self.conversation_history
            if (datetime.fromisoformat(conv["timestamp"]) > cutoff_time and 
                conv.get("interaction_type") == "voice")
        ]
        
        # Calculate statistics
        total_interactions = len(recent_voice_conversations)
        total_audio_files = len(recent_audio)
        
        avg_confidence = 0.0
        total_duration = 0.0
        total_words = 0
        
        if recent_audio:
            confidences = [a["confidence_score"] for a in recent_audio if a["confidence_score"]]
            if confidences:
                avg_confidence = statistics.mean(confidences)
            
            durations = [a.get("audio_duration_seconds", 0) for a in recent_audio]
            total_duration = sum(durations)
            
            total_words = sum(a["word_count"] for a in recent_audio)
        
        # Wake word activity
        recent_wake_words = 0
        for wake_word, stats in self.wake_word_stats.items():
            last_detected = datetime.fromisoformat(stats["last_detected"])
            if last_detected > cutoff_time:
                recent_wake_words += 1
        
        return {
            "time_window_hours": hours,
            "total_voice_interactions": total_interactions,
            "total_audio_files_processed": total_audio_files,
            "total_audio_duration_seconds": total_duration,
            "total_words_transcribed": total_words,
            "average_transcription_confidence": round(avg_confidence, 3),
            "recent_wake_word_activity": recent_wake_words,
            "average_words_per_interaction": round(total_words / max(total_audio_files, 1), 1),
            "interactions_per_hour": round(total_interactions / max(hours, 1), 2)
        }
    
    def get_wake_word_analytics(self) -> Dict[str, Any]:
        """Get comprehensive wake word analytics"""
        analytics = {}
        
        for wake_word, stats in self.wake_word_stats.items():
            confidences = stats["confidence_scores"]
            
            analytics[wake_word] = {
                "total_detections": stats["total_detections"],
                "average_confidence": round(statistics.mean(confidences), 3) if confidences else 0,
                "confidence_std_dev": round(statistics.stdev(confidences), 3) if len(confidences) > 1 else 0,
                "most_active_hour": max(stats["hourly_distribution"], key=stats["hourly_distribution"].get) if stats["hourly_distribution"] else None,
                "most_active_day": max(stats["daily_distribution"], key=stats["daily_distribution"].get) if stats["daily_distribution"] else None,
                "first_detected": stats["first_detected"],
                "last_detected": stats["last_detected"],
                "hourly_pattern": dict(stats["hourly_distribution"]),
                "daily_pattern": dict(stats["daily_distribution"])
            }
        
        return analytics
    
    def get_voice_command_insights(self) -> Dict[str, Any]:
        """Get insights into voice command patterns"""
        if not self.voice_command_patterns:
            return {"message": "No voice command patterns recorded yet"}
        
        # Analyze patterns
        sorted_patterns = sorted(
            self.voice_command_patterns.items(),
            key=lambda x: x[1]["frequency"],
            reverse=True
        )
        
        top_patterns = sorted_patterns[:10]
        
        # Intent distribution
        intent_counts = Counter()
        for pattern_data in self.voice_command_patterns.values():
            intent_counts[pattern_data["intent"]] += pattern_data["frequency"]
        
        # Average metrics
        all_confidences = []
        all_word_counts = []
        
        for pattern_data in self.voice_command_patterns.values():
            if pattern_data["confidence_scores"]:
                all_confidences.extend(pattern_data["confidence_scores"])
            all_word_counts.append(pattern_data["average_word_count"])
        
        return {
            "total_command_patterns": len(self.voice_command_patterns),
            "top_command_patterns": [
                {
                    "pattern": pattern_key,
                    "intent": pattern_data["intent"],
                    "frequency": pattern_data["frequency"],
                    "avg_word_count": round(pattern_data["average_word_count"], 1),
                    "examples": pattern_data["examples"][:3]  # Top 3 examples
                }
                for pattern_key, pattern_data in top_patterns
            ],
            "intent_distribution": dict(intent_counts.most_common()),
            "average_command_confidence": round(statistics.mean(all_confidences), 3) if all_confidences else 0,
            "average_command_length": round(statistics.mean(all_word_counts), 1) if all_word_counts else 0
        }
    
    def get_transcription_quality_report(self) -> Dict[str, Any]:
        """Get detailed transcription quality analysis"""
        if not self.transcription_quality_stats:
            return {"message": "No transcription quality data available"}
        
        stats = self.transcription_quality_stats
        
        # Calculate trends
        recent_scores = stats.get("recent_confidence_scores", [])
        if len(recent_scores) >= 10:
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]
            
            trend = statistics.mean(second_half) - statistics.mean(first_half)
            trend_direction = "improving" if trend > 0.05 else "declining" if trend < -0.05 else "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            "total_transcriptions": stats.get("total_transcriptions", 0),
            "average_confidence": round(stats.get("average_confidence", 0), 3),
            "confidence_std_dev": round(stats.get("confidence_std_dev", 0), 3),
            "average_processing_time": round(stats.get("average_processing_time", 0), 3),
            "average_audio_duration": round(stats.get("average_audio_duration", 0), 3),
            "average_words_per_transcription": round(stats.get("average_words_per_transcription", 0), 1),
            "quality_trend": trend_direction,
            "high_confidence_rate": round(stats.get("high_confidence_rate", 0), 3),
            "low_confidence_rate": round(stats.get("low_confidence_rate", 0), 3)
        }
    
    # ========== Enhanced Original Methods ==========
    
    def get_recent_context(self, limit: int = 3, voice_only: bool = False) -> str:
        """Get recent conversation context with optional voice filtering"""
        if not self.conversation_history:
            return ""
        
        conversations = self.conversation_history
        if voice_only:
            conversations = [
                conv for conv in conversations 
                if conv.get("interaction_type") == "voice"
            ]
        
        recent_turns = conversations[-limit:]
        context_parts = []
        
        for turn in recent_turns:
            interaction_type = turn.get("interaction_type", "text")
            prefix = "🎤" if interaction_type == "voice" else "💬"
            
            context_parts.append(f"{prefix} User: {turn['user'][:100]}...")
            context_parts.append(f"{prefix} Assistant: {turn['assistant'][:100]}...")
        
        return " | ".join(context_parts)
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Enhanced conversation pattern analysis with voice-specific insights"""
        if not self.conversation_history:
            return {"error": "No conversation history available"}
        
        # Separate voice and text interactions
        voice_conversations = [
            conv for conv in self.conversation_history 
            if conv.get("interaction_type") == "voice"
        ]
        text_conversations = [
            conv for conv in self.conversation_history 
            if conv.get("interaction_type") != "voice"
        ]
        
        # Original analysis
        all_user_inputs = [turn['user'].lower() for turn in self.conversation_history]
        
        # Word frequency analysis
        word_counts = {}
        for text in all_user_inputs:
            words = text.split()
            for word in words:
                if len(word) > 3:  # Skip common short words
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_conversations": len(self.conversation_history),
            "voice_conversations": len(voice_conversations),
            "text_conversations": len(text_conversations),
            "voice_interaction_rate": round(len(voice_conversations) / max(len(self.conversation_history), 1), 3),
            "common_topics": [word for word, count in top_words if count > 1],
            "conversation_frequency": self._calculate_conversation_frequency(),
            "most_discussed_objects": self._get_most_discussed_objects(),
            "voice_specific_patterns": self._analyze_voice_patterns()
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Enhanced memory statistics including voice-specific data"""
        base_stats = {
            "conversation_turns": len(self.conversation_history),
            "visual_changes": len(self.visual_context_history),
            "current_objects": len(self.scene_state.get('current_objects', [])),
            "user_preferences": len(self.user_preferences),
            "memory_enabled": self.enable_memory,
            "last_activity": self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }
        
        # Add voice-specific stats
        voice_stats = {
            "audio_metadata_records": len(self.audio_metadata_history),
            "wake_words_tracked": len(self.wake_word_stats),
            "voice_command_patterns": len(self.voice_command_patterns),
            "voice_sessions": len(self.voice_interaction_sessions),
            "audio_metadata_enabled": self.enable_audio_metadata,
            "last_audio_interaction": (
                self.audio_metadata_history[-1]['timestamp'] 
                if self.audio_metadata_history else None
            )
        }
        
        return {**base_stats, **voice_stats}
    
    # ========== Private Helper Methods ==========
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return Path(file_path).stat().st_size
        except (OSError, FileNotFoundError):
            return 0
    
    def _update_transcription_quality_stats(self, metadata: Dict[str, Any]):
        """Update transcription quality statistics"""
        if "total_transcriptions" not in self.transcription_quality_stats:
            self.transcription_quality_stats = {
                "total_transcriptions": 0,
                "confidence_scores": [],
                "processing_times": [],
                "audio_durations": [],
                "word_counts": [],
                "recent_confidence_scores": []
            }
        
        stats = self.transcription_quality_stats
        stats["total_transcriptions"] += 1
        
        # Track metrics
        if metadata.get("confidence_score") is not None:
            stats["confidence_scores"].append(metadata["confidence_score"])
            stats["recent_confidence_scores"].append(metadata["confidence_score"])
            
            # Keep only recent scores for trend analysis
            if len(stats["recent_confidence_scores"]) > 50:
                stats["recent_confidence_scores"] = stats["recent_confidence_scores"][-50:]
        
        if metadata.get("processing_time_seconds") is not None:
            stats["processing_times"].append(metadata["processing_time_seconds"])
        
        if metadata.get("audio_duration_seconds") is not None:
            stats["audio_durations"].append(metadata["audio_duration_seconds"])
        
        stats["word_counts"].append(metadata["word_count"])
        
        # Calculate derived statistics
        if stats["confidence_scores"]:
            stats["average_confidence"] = statistics.mean(stats["confidence_scores"])
            stats["confidence_std_dev"] = statistics.stdev(stats["confidence_scores"]) if len(stats["confidence_scores"]) > 1 else 0
            
            # High/low confidence rates
            high_confidence = sum(1 for score in stats["confidence_scores"] if score > 0.8)
            low_confidence = sum(1 for score in stats["confidence_scores"] if score < 0.5)
            total = len(stats["confidence_scores"])
            
            stats["high_confidence_rate"] = high_confidence / total
            stats["low_confidence_rate"] = low_confidence / total
        
        if stats["processing_times"]:
            stats["average_processing_time"] = statistics.mean(stats["processing_times"])
        
        if stats["audio_durations"]:
            stats["average_audio_duration"] = statistics.mean(stats["audio_durations"])
        
        if stats["word_counts"]:
            stats["average_words_per_transcription"] = statistics.mean(stats["word_counts"])
        
        # Limit history sizes
        for key in ["confidence_scores", "processing_times", "audio_durations", "word_counts"]:
            if len(stats[key]) > 1000:
                stats[key] = stats[key][-1000:]
    
    def _classify_command_intent(self, command_text: str) -> str:
        """Simple intent classification for voice commands"""
        command_lower = command_text.lower()
        
        # Define intent patterns
        intent_patterns = {
            "question": ["what", "how", "why", "when", "where", "who", "?"],
            "request": ["please", "can you", "could you", "would you"],
            "command": ["play", "stop", "start", "open", "close", "turn"],
            "information": ["tell me", "show me", "explain", "describe"],
            "navigation": ["go to", "navigate", "find", "search"],
            "control": ["volume", "brightness", "temperature", "settings"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in command_lower for pattern in patterns):
                return intent
        
        return "general"
    
    def _analyze_voice_patterns(self) -> Dict[str, Any]:
        """Analyze voice-specific conversation patterns"""
        voice_conversations = [
            conv for conv in self.conversation_history 
            if conv.get("interaction_type") == "voice"
        ]
        
        if not voice_conversations:
            return {"message": "No voice conversations available"}
        
        # Analyze voice interaction timing
        timestamps = [datetime.fromisoformat(conv["timestamp"]) for conv in voice_conversations]
        
        # Hour distribution
        hour_distribution = defaultdict(int)
        for ts in timestamps:
            hour_distribution[ts.hour] += 1
        
        # Day distribution
        day_distribution = defaultdict(int)
        for ts in timestamps:
            day_distribution[ts.strftime("%A")] += 1
        
        # Average conversation length (in words)
        voice_inputs = [conv["user"] for conv in voice_conversations]
        avg_input_length = statistics.mean([len(text.split()) for text in voice_inputs])
        
        return {
            "total_voice_conversations": len(voice_conversations),
            "average_input_length_words": round(avg_input_length, 1),
            "most_active_hour": max(hour_distribution, key=hour_distribution.get) if hour_distribution else None,
            "most_active_day": max(day_distribution, key=day_distribution.get) if day_distribution else None,
            "hourly_distribution": dict(hour_distribution),
            "daily_distribution": dict(day_distribution)
        }
    
    def _calculate_conversation_frequency(self) -> Dict[str, int]:
        """Calculate conversation frequency by time period (inherited from original)"""
        now = datetime.now()
        periods = {
            "last_hour": now - timedelta(hours=1),
            "last_day": now - timedelta(days=1),
            "last_week": now - timedelta(weeks=1)
        }
        
        frequency = {}
        for period_name, cutoff in periods.items():
            count = sum(1 for turn in self.conversation_history 
                       if datetime.fromisoformat(turn['timestamp']) > cutoff)
            frequency[period_name] = count
        
        return frequency
    
    def _get_most_discussed_objects(self) -> List[str]:
        """Find most frequently discussed objects (inherited from original)"""
        object_mentions = {}
        
        for turn in self.conversation_history:
            context_objects = turn.get('context_objects', [])
            for obj in context_objects:
                object_mentions[obj] = object_mentions.get(obj, 0) + 1
        
        # Return top 5 most mentioned objects
        top_objects = sorted(object_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        return [obj for obj, count in top_objects]
    
    def save_user_preference(self, key: str, value: Any):
        """Save user preference (inherited from original)"""
        self.user_preferences[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self._save_preferences()
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference (inherited from original)"""
        pref = self.user_preferences.get(key)
        return pref['value'] if pref else default
    
    # ========== Persistence Methods ==========
    
    def _load_persistent_memory(self):
        """Load memory from persistent storage (enhanced)"""
        try:
            # Load original memory types
            if self.conversation_file.exists():
                with open(self.conversation_file, 'r') as f:
                    self.conversation_history = json.load(f)
            
            if self.visual_file.exists():
                with open(self.visual_file, 'r') as f:
                    data = json.load(f)
                    self.visual_context_history = data.get('history', [])
                    self.scene_state = data.get('current_state', {})
            
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    self.user_preferences = json.load(f)
            
            # Load voice-specific memory types
            if self.audio_metadata_file.exists():
                with open(self.audio_metadata_file, 'r') as f:
                    self.audio_metadata_history = json.load(f)
            
            if self.wake_word_file.exists():
                with open(self.wake_word_file, 'r') as f:
                    self.wake_word_stats = json.load(f)
            
            if self.voice_patterns_file.exists():
                with open(self.voice_patterns_file, 'r') as f:
                    self.voice_command_patterns = json.load(f)
            
            if self.transcription_stats_file.exists():
                with open(self.transcription_stats_file, 'r') as f:
                    self.transcription_quality_stats = json.load(f)
            
            if self.voice_sessions_file.exists():
                with open(self.voice_sessions_file, 'r') as f:
                    self.voice_interaction_sessions = json.load(f)
                    
        except Exception as e:
            print(f"⚠️ Memory load error: {e}")
    
    def _save_conversations(self):
        """Save conversations to persistent storage (inherited from original)"""
        try:
            with open(self.conversation_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Conversation save error: {e}")
    
    def _save_visual_context(self):
        """Save visual context to persistent storage (inherited from original)"""
        try:
            data = {
                'current_state': self.scene_state,
                'history': self.visual_context_history
            }
            with open(self.visual_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Visual context save error: {e}")
    
    def _save_preferences(self):
        """Save user preferences to persistent storage (inherited from original)"""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            print(f"⚠️ Preferences save error: {e}")
    
    def _save_audio_metadata(self):
        """Save audio metadata to persistent storage"""
        try:
            with open(self.audio_metadata_file, 'w') as f:
                json.dump(self.audio_metadata_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Audio metadata save error: {e}")
    
    def _save_wake_word_stats(self):
        """Save wake word statistics to persistent storage"""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            serializable_stats = {}
            for wake_word, stats in self.wake_word_stats.items():
                serializable_stats[wake_word] = {
                    **stats,
                    "hourly_distribution": dict(stats["hourly_distribution"]),
                    "daily_distribution": dict(stats["daily_distribution"])
                }
            
            with open(self.wake_word_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
        except Exception as e:
            print(f"⚠️ Wake word stats save error: {e}")
    
    def _save_voice_patterns(self):
        """Save voice command patterns to persistent storage"""
        try:
            with open(self.voice_patterns_file, 'w') as f:
                json.dump(self.voice_command_patterns, f, indent=2)
        except Exception as e:
            print(f"⚠️ Voice patterns save error: {e}")
    
    def _save_transcription_stats(self):
        """Save transcription quality statistics to persistent storage"""
        try:
            with open(self.transcription_stats_file, 'w') as f:
                json.dump(self.transcription_quality_stats, f, indent=2)
        except Exception as e:
            print(f"⚠️ Transcription stats save error: {e}")
    
    def _save_voice_sessions(self):
        """Save voice interaction sessions to persistent storage"""
        try:
            with open(self.voice_sessions_file, 'w') as f:
                json.dump(self.voice_interaction_sessions, f, indent=2)
        except Exception as e:
            print(f"⚠️ Voice sessions save error: {e}")
    
    def save_all_memory(self):
        """Save all memory to persistent storage (enhanced)"""
        print("💾 Saving all memory to disk...")
        self._save_conversations()
        self._save_visual_context()
        self._save_preferences()
        self._save_audio_metadata()
        self._save_wake_word_stats()
        self._save_voice_patterns()
        self._save_transcription_stats()
        self._save_voice_sessions()
        print("✅ All memory saved")
    
    def clear_memory(self, keep_preferences: bool = True, clear_voice_data: bool = True):
        """Clear memory with options for voice data (enhanced)"""
        print("🗑️ Clearing memory...")
        
        # Clear original memory types
        self.conversation_history = []
        self.visual_context_history = []
        self.scene_state = {}
        
        if not keep_preferences:
            self.user_preferences = {}
        
        # Clear voice-specific memory
        if clear_voice_data:
            self.audio_metadata_history = []
            self.wake_word_stats = {}
            self.voice_command_patterns = {}
            self.transcription_quality_stats = {}
            self.voice_interaction_sessions = []
        
        # Clear persistent files
        files_to_clear = [self.conversation_file, self.visual_file]
        if clear_voice_data:
            files_to_clear.extend([
                self.audio_metadata_file,
                self.wake_word_file,
                self.voice_patterns_file,
                self.transcription_stats_file,
                self.voice_sessions_file
            ])
        
        for file_path in files_to_clear:
            if file_path.exists():
                file_path.unlink()
        
        if not keep_preferences and self.preferences_file.exists():
            self.preferences_file.unlink()
        
        print("✅ Memory cleared")

# Demo usage and testing
if __name__ == "__main__":
    # Demo the enhanced voice context memory system
    config = {
        'enable_context_memory': True,
        'enable_audio_metadata': True,
        'conversation_memory_limit': 10,
        'audio_history_limit': 50
    }
    
    voice_memory = VoiceContextMemory(config)
    
    print("🧠🎤 Voice Context Memory Demo")
    print("=" * 40)
    
    # Demo voice conversation with audio metadata
    audio_metadata = {
        "confidence_score": 0.87,
        "processing_time_seconds": 2.3,
        "audio_duration_seconds": 4.5
    }
    
    voice_memory.add_conversation_turn(
        "What do you see in the room?", 
        "I can see a person sitting at a desk with a laptop",
        audio_metadata=audio_metadata
    )
    
    # Demo audio metadata tracking
    transcription_result = {
        "text": "What do you see in the room",
        "language": "en",
        "segments": [{"start": 0, "end": 4.5, "text": "What do you see in the room"}]
    }
    
    voice_memory.add_audio_metadata(
        "/tmp/test_audio.wav",
        transcription_result,
        confidence_score=0.87,
        processing_time=2.3
    )
    
    # Demo wake word tracking
    voice_memory.track_wake_word_detection("hey assistant", 0.92)
    voice_memory.track_wake_word_detection("hey assistant", 0.88)
    
    # Demo voice command pattern analysis
    voice_memory.analyze_voice_command_patterns("What do you see in the room", "question", 0.87)
    voice_memory.analyze_voice_command_patterns("Show me the weather", "request", 0.91)
    
    # Demo voice session tracking
    session_id = voice_memory.start_voice_session("demo")
    time.sleep(1)  # Simulate session duration
    voice_memory.end_voice_session(session_id)
    
    # Show enhanced analytics
    print("\n📊 Voice Interaction Summary (last 24h):")
    print(json.dumps(voice_memory.get_voice_interaction_summary(), indent=2))
    
    print("\n🎯 Wake Word Analytics:")
    print(json.dumps(voice_memory.get_wake_word_analytics(), indent=2))
    
    print("\n🗣️ Voice Command Insights:")
    print(json.dumps(voice_memory.get_voice_command_insights(), indent=2))
    
    print("\n📈 Transcription Quality Report:")
    print(json.dumps(voice_memory.get_transcription_quality_report(), indent=2))
    
    print("\n🧠 Enhanced Memory Stats:")
    print(json.dumps(voice_memory.get_memory_stats(), indent=2))
    
    print("\n💾 Saving all memory...")
    voice_memory.save_all_memory()
    
    print("✅ Voice Context Memory Demo complete!")