#!/usr/bin/env python3
"""
Voice Memory Manager - Central coordination point for all voice memory operations

This class coordinates between VoiceContextMemory and VoiceConversationMemory
to provide a unified interface for voice command processing, memory fusion,
pattern analysis, and learning capabilities.

Features:
- Unified interface for both memory systems
- Memory fusion for comprehensive context
- Pattern analysis and learning
- Configuration management
- Error handling and logging
- Performance optimization
- Voice interaction lifecycle management
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .voice_context_memory import VoiceContextMemory
from .voice_conversation_memory import VoiceConversationMemory, create_voice_memory


class VoiceMemoryManager:
    """Central coordination point for all voice memory operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Voice Memory Manager with both memory systems
        
        Args:
            config: Configuration dictionary for memory systems
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Initialize memory systems
        self.context_memory = None
        self.conversation_memory = None
        self._init_memory_systems()
        
        # State tracking
        self.current_session_id = None
        self._closed = False
        self.active_interactions = {}
        self.performance_stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "average_response_time": 0.0,
            "memory_fusion_calls": 0,
            "pattern_learning_events": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="voice_memory")
        
        self.logger.info("🧠🎤 Voice Memory Manager initialized successfully")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            # Context Memory settings
            'enable_context_memory': True,
            'enable_audio_metadata': True,
            'conversation_memory_limit': 100,
            'audio_history_limit': 500,
            
            # Conversation Memory settings
            'conversation_db_path': 'data/voice_conversations.db',
            'conversation_cache_size': 200,
            
            # Manager settings
            'enable_memory_fusion': True,
            'enable_pattern_learning': True,
            'enable_async_processing': True,
            'auto_save_interval': 300,  # 5 minutes
            'performance_tracking': True,
            
            # Learning settings
            'pattern_learning_threshold': 3,
            'confidence_threshold': 0.7,
            'wake_word_learning_enabled': True,
            'transcription_quality_tracking': True,
            
            # Performance settings
            'max_context_window': 10,
            'memory_cleanup_interval': 3600,  # 1 hour
            'enable_memory_compression': True,
            
            # Error handling
            'fallback_to_single_memory': True,
            'retry_failed_operations': True,
            'max_retry_attempts': 3
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _init_memory_systems(self):
        """Initialize both memory systems with error handling"""
        try:
            # Initialize Context Memory
            if self.config.get('enable_context_memory', True):
                self.context_memory = VoiceContextMemory(self.config)
                self.logger.info("✅ Voice Context Memory initialized")
            
            # Initialize Conversation Memory
            conversation_config = {
                'db_path': self.config.get('conversation_db_path', 'data/voice_conversations.db'),
                'cache_size': self.config.get('conversation_cache_size', 200)
            }
            self.conversation_memory = create_voice_memory(conversation_config['db_path'])
            self.logger.info("✅ Voice Conversation Memory initialized")
            
            # Set current session from conversation memory
            if self.conversation_memory:
                self.current_session_id = self.conversation_memory.current_session_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize memory systems: {e}")
            if not self.config.get('fallback_to_single_memory', True):
                raise
            self.logger.warning("⚠️ Continuing with limited memory functionality")
    
    # ========== Unified Memory Interface ==========
    
    def add_voice_interaction(self,
                            user_input: str,
                            assistant_response: str,
                            audio_metadata: Optional[Dict] = None,
                            agent_used: Optional[str] = None,
                            command_type: Optional[str] = None,
                            success: bool = True,
                            processing_times: Optional[Dict] = None,
                            audio_quality: Optional[Dict] = None,
                            context_objects: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add a complete voice interaction to both memory systems
        
        Args:
            user_input: User's voice input (transcribed)
            assistant_response: Assistant's response
            audio_metadata: Audio file and transcription metadata
            agent_used: Which agent processed the command
            command_type: Type of command (question, request, etc.)
            success: Whether the interaction was successful
            processing_times: Breakdown of processing times
            audio_quality: Audio quality metrics
            context_objects: Objects detected in the scene
            
        Returns:
            Dictionary with memory operation results
        """
        start_time = time.time()
        results = {
            "context_memory_id": None,
            "conversation_id": None,
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "errors": []
        }
        
        with self._lock:
            try:
                # Add to Context Memory
                if self.context_memory:
                    try:
                        self.context_memory.add_conversation_turn(
                            user_input=user_input,
                            assistant_response=assistant_response,
                            audio_metadata=audio_metadata
                        )
                        
                        # Add audio metadata if provided
                        if audio_metadata and audio_metadata.get('audio_file_path'):
                            transcription_result = {
                                'text': user_input,
                                'language': audio_metadata.get('language', 'en'),
                                'segments': audio_metadata.get('segments', [])
                            }
                            
                            self.context_memory.add_audio_metadata(
                                audio_file_path=audio_metadata['audio_file_path'],
                                transcription_result=transcription_result,
                                confidence_score=audio_metadata.get('confidence_score'),
                                processing_time=audio_metadata.get('processing_time')
                            )
                        
                        results["context_memory_success"] = True
                        
                    except Exception as e:
                        self.logger.error(f"Context memory error: {e}")
                        results["errors"].append(f"Context memory: {e}")
                
                # Add to Conversation Memory
                if self.conversation_memory:
                    try:
                        conversation_id = self.conversation_memory.add_voice_conversation(
                            user_input=user_input,
                            assistant_response=assistant_response,
                            agent_used=agent_used,
                            command_type=command_type,
                            success=success,
                            response_time=time.time() - start_time,
                            metadata=audio_metadata,
                            audio_file_path=audio_metadata.get('audio_file_path') if audio_metadata else None,
                            audio_duration=audio_metadata.get('audio_duration') if audio_metadata else None,
                            transcription_confidence=audio_metadata.get('confidence_score') if audio_metadata else None,
                            wake_word_detected=audio_metadata.get('wake_word_detected', False) if audio_metadata else False,
                            wake_word_confidence=audio_metadata.get('wake_word_confidence') if audio_metadata else None,
                            noise_level=audio_quality.get('noise_level') if audio_quality else None,
                            voice_activity_ratio=audio_quality.get('voice_activity_ratio') if audio_quality else None,
                            # Processing times
                            wake_word_detection_time=processing_times.get('wake_word_time') if processing_times else None,
                            transcription_time=processing_times.get('transcription_time') if processing_times else None,
                            llm_processing_time=processing_times.get('llm_time') if processing_times else None,
                            tts_generation_time=processing_times.get('tts_time') if processing_times else None,
                            total_processing_time=processing_times.get('total_time') if processing_times else None,
                            # Audio quality metrics
                            signal_to_noise_ratio=audio_quality.get('snr') if audio_quality else None,
                            audio_clarity_score=audio_quality.get('clarity') if audio_quality else None,
                            background_noise_type=audio_quality.get('noise_type') if audio_quality else None,
                            voice_pitch_range=audio_quality.get('pitch_range') if audio_quality else None,
                            speaking_rate=audio_quality.get('speaking_rate') if audio_quality else None
                        )
                        
                        results["conversation_id"] = conversation_id
                        results["conversation_memory_success"] = True
                        
                        # Add transcription quality data
                        if audio_metadata and audio_metadata.get('confidence_score') is not None:
                            self.conversation_memory.add_transcription_quality(
                                conversation_id=conversation_id,
                                confidence_score=audio_metadata['confidence_score'],
                                word_count=len(user_input.split()),
                                uncertain_words=audio_metadata.get('uncertain_words', 0),
                                language_detected=audio_metadata.get('language', 'en')
                            )
                        
                    except Exception as e:
                        self.logger.error(f"Conversation memory error: {e}")
                        results["errors"].append(f"Conversation memory: {e}")
                
                # Learn patterns if enabled
                if self.config.get('enable_pattern_learning', True):
                    self._learn_voice_patterns_async(user_input, command_type, audio_metadata)
                
                # Update performance stats
                self._update_performance_stats(success, time.time() - start_time)
                
                results["success"] = len(results["errors"]) == 0
                
            except Exception as e:
                self.logger.error(f"Voice interaction storage failed: {e}")
                results["success"] = False
                results["errors"].append(f"General error: {e}")
        
        return results
    
    def add_wake_word_event(self,
                           wake_word: str,
                           confidence: float,
                           conversation_id: Optional[int] = None,
                           false_positive: bool = False,
                           detection_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Record a wake word detection event in both memory systems
        
        Args:
            wake_word: The detected wake word
            confidence: Detection confidence
            conversation_id: Associated conversation ID
            false_positive: Whether this was a false positive
            detection_context: Additional context about the detection
            
        Returns:
            Results from both memory systems
        """
        results = {
            "context_memory_success": False,
            "conversation_memory_success": False,
            "errors": []
        }
        
        with self._lock:
            try:
                # Add to Context Memory
                if self.context_memory:
                    try:
                        self.context_memory.track_wake_word_detection(
                            wake_word=wake_word,
                            confidence=confidence,
                            audio_file_path=detection_context.get('audio_file_path') if detection_context else None
                        )
                        results["context_memory_success"] = True
                    except Exception as e:
                        results["errors"].append(f"Context memory: {e}")
                
                # Add to Conversation Memory
                if self.conversation_memory:
                    try:
                        wake_event_id = self.conversation_memory.add_wake_word_event(
                            wake_word=wake_word,
                            confidence=confidence,
                            conversation_id=conversation_id,
                            false_positive=false_positive,
                            detection_latency=detection_context.get('detection_latency') if detection_context else None,
                            background_context=detection_context.get('background_context') if detection_context else None
                        )
                        results["wake_event_id"] = wake_event_id
                        results["conversation_memory_success"] = True
                    except Exception as e:
                        results["errors"].append(f"Conversation memory: {e}")
                
            except Exception as e:
                self.logger.error(f"Wake word event recording failed: {e}")
                results["errors"].append(f"General error: {e}")
        
        return results
    
    def update_visual_context(self, vision_result: Dict[str, Any]) -> bool:
        """
        Update visual context in context memory
        
        Args:
            vision_result: Results from vision system
            
        Returns:
            Success status
        """
        if not self.context_memory:
            return False
        
        try:
            self.context_memory.update_visual_context(vision_result)
            return True
        except Exception as e:
            self.logger.error(f"Visual context update failed: {e}")
            return False
    
    # ========== Memory Fusion Methods ==========
    
    def get_fused_context(self,
                         context_window_size: int = None,
                         include_audio_metadata: bool = True,
                         include_visual_context: bool = True,
                         voice_only: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive context by fusing both memory systems
        
        Args:
            context_window_size: Number of recent interactions to include
            include_audio_metadata: Whether to include audio metadata
            include_visual_context: Whether to include visual context
            voice_only: Only include voice interactions
            
        Returns:
            Fused context dictionary
        """
        if context_window_size is None:
            context_window_size = self.config.get('max_context_window', 10)
        
        fused_context = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "context_window_size": context_window_size,
            "conversation_history": [],
            "visual_context": {},
            "voice_analytics": {},
            "audio_quality_summary": {},
            "pattern_insights": {},
            "memory_stats": {}
        }
        
        try:
            with self._lock:
                # Get conversation history from conversation memory
                if self.conversation_memory:
                    recent_conversations = self.conversation_memory.get_recent(
                        limit=context_window_size,
                        session_only=False
                    )
                    
                    # Filter for voice only if requested
                    if voice_only:
                        recent_conversations = [
                            conv for conv in recent_conversations
                            if conv.get('transcription_confidence') is not None
                        ]
                    
                    fused_context["conversation_history"] = recent_conversations
                
                # Get visual context from context memory
                if self.context_memory and include_visual_context:
                    fused_context["visual_context"] = {
                        "current_scene": self.context_memory.scene_state,
                        "recent_changes": self.context_memory.visual_context_history[-5:] if self.context_memory.visual_context_history else []
                    }
                
                # Get voice-specific analytics
                if include_audio_metadata:
                    # From context memory
                    if self.context_memory:
                        fused_context["voice_analytics"]["context_summary"] = self.context_memory.get_voice_interaction_summary(hours=24)
                        fused_context["voice_analytics"]["wake_word_analytics"] = self.context_memory.get_wake_word_analytics()
                        fused_context["voice_analytics"]["command_insights"] = self.context_memory.get_voice_command_insights()
                        fused_context["voice_analytics"]["transcription_quality"] = self.context_memory.get_transcription_quality_report()
                    
                    # From conversation memory
                    if self.conversation_memory:
                        fused_context["voice_analytics"]["detailed_analytics"] = self.conversation_memory.get_voice_analytics(days=7)
                        fused_context["voice_analytics"]["quality_report"] = self.conversation_memory.get_transcription_quality_report(days=7)
                
                # Get pattern insights
                if self.config.get('enable_pattern_learning', True):
                    fused_context["pattern_insights"] = self._get_pattern_insights()
                
                # Memory system stats
                fused_context["memory_stats"] = self._get_memory_fusion_stats()
                
                # Update fusion call counter
                self.performance_stats["memory_fusion_calls"] += 1
                
        except Exception as e:
            self.logger.error(f"Memory fusion failed: {e}")
            fused_context["error"] = str(e)
        
        return fused_context
    
    def get_contextual_response_hints(self, user_input: str) -> Dict[str, Any]:
        """
        Get contextual hints for generating better responses
        
        Args:
            user_input: Current user input
            
        Returns:
            Contextual hints and suggestions
        """
        hints = {
            "similar_past_interactions": [],
            "relevant_visual_context": {},
            "suggested_response_patterns": [],
            "audio_quality_considerations": {},
            "user_preferences": {}
        }
        
        try:
            # Search for similar past interactions
            if self.conversation_memory:
                similar_convs = self.conversation_memory.search_voice_conversations(
                    query=user_input,
                    limit=5,
                    min_confidence=0.7
                )
                hints["similar_past_interactions"] = similar_convs
            
            # Get relevant visual context
            if self.context_memory:
                current_objects = self.context_memory.scene_state.get('current_objects', [])
                if current_objects:
                    hints["relevant_visual_context"] = {
                        "current_objects": current_objects,
                        "confidence_scores": self.context_memory.scene_state.get('confidence_scores', {}),
                        "last_updated": self.context_memory.scene_state.get('last_updated')
                    }
                
                # Get user preferences
                hints["user_preferences"] = self.context_memory.user_preferences
            
            # Analyze input for response patterns
            if self.config.get('enable_pattern_learning', True):
                hints["suggested_response_patterns"] = self._analyze_input_patterns(user_input)
            
        except Exception as e:
            self.logger.error(f"Failed to get contextual hints: {e}")
            hints["error"] = str(e)
        
        return hints
    
    # ========== Pattern Analysis and Learning ==========
    
    def _learn_voice_patterns_async(self,
                                  user_input: str,
                                  command_type: Optional[str],
                                  audio_metadata: Optional[Dict]):
        """Asynchronously learn voice patterns"""
        if not self.config.get('enable_async_processing', True):
            return self._learn_voice_patterns_sync(user_input, command_type, audio_metadata)
        
        future = self.executor.submit(
            self._learn_voice_patterns_sync,
            user_input, command_type, audio_metadata
        )
        return future
    
    def _learn_voice_patterns_sync(self,
                                 user_input: str,
                                 command_type: Optional[str],
                                 audio_metadata: Optional[Dict]):
        """Learn voice patterns synchronously"""
        try:
            # Learn in context memory
            if self.context_memory:
                confidence = audio_metadata.get('confidence_score') if audio_metadata else None
                self.context_memory.analyze_voice_command_patterns(
                    command_text=user_input,
                    intent=command_type,
                    confidence=confidence
                )
            
            # Learn in conversation memory
            if self.conversation_memory and command_type:
                pattern_id = self.conversation_memory.learn_voice_pattern(
                    pattern_type=command_type,
                    pattern_text=user_input,
                    user_intent=command_type,
                    context_tags=self._extract_context_tags(user_input)
                )
                self.logger.debug(f"Learned pattern {pattern_id}")
            
            self.performance_stats["pattern_learning_events"] += 1
            
        except Exception as e:
            self.logger.error(f"Pattern learning failed: {e}")
    
    def _extract_context_tags(self, text: str) -> List[str]:
        """Extract context tags from text"""
        tags = []
        text_lower = text.lower()
        
        # Simple keyword-based tagging
        tag_keywords = {
            'time': ['time', 'clock', 'when', 'schedule'],
            'weather': ['weather', 'temperature', 'rain', 'sunny'],
            'control': ['turn', 'switch', 'control', 'adjust'],
            'information': ['what', 'how', 'why', 'explain', 'tell'],
            'navigation': ['go', 'navigate', 'find', 'location'],
            'media': ['play', 'music', 'video', 'volume']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _analyze_input_patterns(self, user_input: str) -> List[Dict]:
        """Analyze input patterns for response suggestions"""
        patterns = []
        
        try:
            # Get similar patterns from conversation memory
            if self.conversation_memory:
                # This would typically involve more sophisticated pattern matching
                # For now, we'll use simple heuristics
                input_lower = user_input.lower()
                
                if any(word in input_lower for word in ['what', 'how', 'why', 'when', 'where']):
                    patterns.append({
                        "type": "question",
                        "confidence": 0.8,
                        "suggestion": "Provide informative answer with context"
                    })
                
                if any(word in input_lower for word in ['please', 'can you', 'could you']):
                    patterns.append({
                        "type": "polite_request",
                        "confidence": 0.9,
                        "suggestion": "Acknowledge politeness and fulfill request"
                    })
                
                if any(word in input_lower for word in ['turn', 'switch', 'control']):
                    patterns.append({
                        "type": "control_command",
                        "confidence": 0.7,
                        "suggestion": "Execute control action and provide confirmation"
                    })
        
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
        
        return patterns
    
    def _get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights from learned patterns"""
        insights = {
            "top_command_types": [],
            "success_patterns": [],
            "improvement_suggestions": []
        }
        
        try:
            # Get insights from context memory
            if self.context_memory:
                command_insights = self.context_memory.get_voice_command_insights()
                if "top_command_patterns" in command_insights:
                    insights["top_command_types"] = command_insights["top_command_patterns"][:5]
            
            # Get optimization suggestions from conversation memory
            if self.conversation_memory:
                suggestions = self.conversation_memory.optimize_voice_patterns(
                    min_frequency=self.config.get('pattern_learning_threshold', 3)
                )
                insights["improvement_suggestions"] = suggestions[:5]
        
        except Exception as e:
            self.logger.error(f"Pattern insights failed: {e}")
            insights["error"] = str(e)
        
        return insights
    
    # ========== Search and Retrieval ==========
    
    def search_voice_interactions(self,
                                query: str,
                                limit: int = 10,
                                search_scope: str = "both",
                                filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search across both memory systems
        
        Args:
            query: Search query
            limit: Maximum results
            search_scope: "both", "context", or "conversation"
            filters: Additional filters (confidence, time range, etc.)
            
        Returns:
            Search results from both systems
        """
        results = {
            "query": query,
            "search_scope": search_scope,
            "context_results": [],
            "conversation_results": [],
            "total_results": 0
        }
        
        try:
            # Search conversation memory
            if search_scope in ["both", "conversation"] and self.conversation_memory:
                min_confidence = filters.get('min_confidence') if filters else None
                wake_word_only = filters.get('wake_word_only', False) if filters else False
                
                conv_results = self.conversation_memory.search_voice_conversations(
                    query=query,
                    limit=limit,
                    min_confidence=min_confidence,
                    wake_word_only=wake_word_only
                )
                results["conversation_results"] = conv_results
            
            # Search context memory (basic text search)
            if search_scope in ["both", "context"] and self.context_memory:
                # Simple search through conversation history
                context_matches = []
                for conv in self.context_memory.conversation_history:
                    if query.lower() in conv.get('user', '').lower() or \
                       query.lower() in conv.get('assistant', '').lower():
                        context_matches.append({
                            "timestamp": conv.get('timestamp'),
                            "user": conv.get('user'),
                            "assistant": conv.get('assistant'),
                            "interaction_type": conv.get('interaction_type', 'text')
                        })
                
                results["context_results"] = context_matches[:limit]
            
            results["total_results"] = len(results["conversation_results"]) + len(results["context_results"])
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_recent_interactions(self,
                              limit: int = 10,
                              voice_only: bool = False,
                              include_metadata: bool = True) -> List[Dict]:
        """
        Get recent interactions with optional filtering
        
        Args:
            limit: Number of interactions to return
            voice_only: Only return voice interactions
            include_metadata: Include audio metadata
            
        Returns:
            List of recent interactions
        """
        try:
            if self.conversation_memory:
                interactions = self.conversation_memory.get_recent(limit=limit)
                
                if voice_only:
                    interactions = [
                        i for i in interactions 
                        if i.get('transcription_confidence') is not None
                    ]
                
                if not include_metadata:
                    # Remove metadata to reduce payload size
                    for interaction in interactions:
                        interaction.pop('metadata', None)
                
                return interactions
            
            elif self.context_memory:
                # Fallback to context memory
                context_str = self.context_memory.get_recent_context(limit=limit, voice_only=voice_only)
                # Convert to structured format (simplified)
                return [{"context": context_str}]
            
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get recent interactions: {e}")
            return []
    
    # ========== Analytics and Reporting ==========
    
    def get_comprehensive_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive analytics from both memory systems
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Comprehensive analytics report
        """
        analytics = {
            "analysis_period_days": days,
            "timestamp": datetime.now().isoformat(),
            "voice_interaction_summary": {},
            "transcription_quality": {},
            "wake_word_performance": {},
            "conversation_patterns": {},
            "audio_quality_trends": {},
            "system_performance": {},
            "memory_efficiency": {},
            "recommendations": []
        }
        
        try:
            # Get analytics from conversation memory
            if self.conversation_memory:
                voice_analytics = self.conversation_memory.get_voice_analytics(days=days)
                analytics["voice_interaction_summary"] = voice_analytics.get("basic_stats", {})
                analytics["wake_word_performance"] = voice_analytics.get("wake_word_performance", {})
                analytics["audio_quality_trends"] = voice_analytics.get("audio_quality", {})
                
                quality_report = self.conversation_memory.get_transcription_quality_report(days=days)
                analytics["transcription_quality"] = quality_report
            
            # Get analytics from context memory
            if self.context_memory:
                context_summary = self.context_memory.get_voice_interaction_summary(hours=days*24)
                analytics["voice_interaction_summary"].update({
                    "context_memory_stats": context_summary
                })
                
                pattern_analysis = self.context_memory.analyze_conversation_patterns()
                analytics["conversation_patterns"] = pattern_analysis
            
            # System performance stats
            analytics["system_performance"] = self.performance_stats.copy()
            
            # Memory efficiency stats
            analytics["memory_efficiency"] = self._get_memory_fusion_stats()
            
            # Generate recommendations
            analytics["recommendations"] = self._generate_recommendations(analytics)
            
        except Exception as e:
            self.logger.error(f"Analytics generation failed: {e}")
            analytics["error"] = str(e)
        
        return analytics
    
    def _get_memory_fusion_stats(self) -> Dict[str, Any]:
        """Get memory fusion efficiency statistics"""
        stats = {
            "context_memory_available": self.context_memory is not None,
            "conversation_memory_available": self.conversation_memory is not None,
            "current_session_id": self.current_session_id,
            "memory_fusion_calls": self.performance_stats.get("memory_fusion_calls", 0),
            "active_interactions": len(self.active_interactions)
        }
        
        try:
            if self.context_memory:
                context_stats = self.context_memory.get_memory_stats()
                stats["context_memory_stats"] = context_stats
            
            if self.conversation_memory:
                # Get database size info if available
                stats["conversation_memory_health"] = True
        
        except Exception as e:
            self.logger.error(f"Memory stats collection failed: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def _generate_recommendations(self, analytics: Dict[str, Any]) -> List[Dict]:
        """Generate recommendations based on analytics"""
        recommendations = []
        
        try:
            # Transcription quality recommendations
            transcription_quality = analytics.get("transcription_quality", {})
            if transcription_quality:
                summary = transcription_quality.get("summary", {})
                avg_confidence = summary.get("overall_avg_confidence", 0)
                
                if avg_confidence < 0.7:
                    recommendations.append({
                        "type": "audio_quality",
                        "priority": "high",
                        "message": f"Average transcription confidence is low ({avg_confidence:.2f}). Consider improving audio setup.",
                        "suggestions": [
                            "Check microphone positioning",
                            "Reduce background noise",
                            "Adjust microphone sensitivity"
                        ]
                    })
                
                correction_rate = summary.get("correction_rate", 0)
                if correction_rate > 0.1:
                    recommendations.append({
                        "type": "transcription",
                        "priority": "medium",
                        "message": f"High correction rate ({correction_rate:.2f}). Consider voice training.",
                        "suggestions": [
                            "Speak more clearly",
                            "Use consistent pronunciation",
                            "Train wake word recognition"
                        ]
                    })
            
            # Wake word performance recommendations
            wake_word_perf = analytics.get("wake_word_performance", {})
            if wake_word_perf:
                false_positive_rate = wake_word_perf.get("false_positive_rate", 0)
                if false_positive_rate > 0.05:
                    recommendations.append({
                        "type": "wake_word",
                        "priority": "medium",
                        "message": f"High false positive rate ({false_positive_rate:.2f}) for wake word detection.",
                        "suggestions": [
                            "Adjust wake word sensitivity",
                            "Train with different voice samples",
                            "Consider alternative wake words"
                        ]
                    })
            
            # Performance recommendations
            system_perf = analytics.get("system_performance", {})
            avg_response_time = system_perf.get("average_response_time", 0)
            if avg_response_time > 3.0:
                recommendations.append({
                    "type": "performance",
                    "priority": "medium",
                    "message": f"Slow average response time ({avg_response_time:.2f}s).",
                    "suggestions": [
                        "Optimize processing pipeline",
                        "Consider hardware upgrades",
                        "Enable async processing"
                    ]
                })
        
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    # ========== Configuration and Lifecycle Management ==========
    
    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """
        Update configuration dynamically
        
        Args:
            new_config: New configuration values
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                self.config.update(new_config)
                self.logger.info("Configuration updated successfully")
                return True
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return False
    
    def save_all_memory(self) -> Dict[str, bool]:
        """Save all memory to persistent storage"""
        results = {
            "context_memory": False,
            "conversation_memory": False
        }
        
        try:
            if self.context_memory:
                self.context_memory.save_all_memory()
                results["context_memory"] = True
            
            if self.conversation_memory:
                # Conversation memory auto-saves, but we can force a commit
                results["conversation_memory"] = True
                
        except Exception as e:
            self.logger.error(f"Memory save failed: {e}")
        
        return results
    
    def clear_memory(self,
                    clear_context: bool = True,
                    clear_conversation: bool = True,
                    keep_preferences: bool = True) -> Dict[str, bool]:
        """
        Clear memory with granular control
        
        Args:
            clear_context: Clear context memory
            clear_conversation: Clear conversation memory  
            keep_preferences: Keep user preferences
            
        Returns:
            Results of clear operations
        """
        results = {
            "context_cleared": False,
            "conversation_cleared": False
        }
        
        try:
            with self._lock:
                if clear_context and self.context_memory:
                    self.context_memory.clear_memory(
                        keep_preferences=keep_preferences,
                        clear_voice_data=True
                    )
                    results["context_cleared"] = True
                
                if clear_conversation and self.conversation_memory:
                    # Note: VoiceConversationMemory doesn't have a clear method
                    # This would need to be implemented or we recreate the instance
                    self.logger.warning("Conversation memory clearing not implemented")
                
                # Reset performance stats
                self.performance_stats = {
                    "total_interactions": 0,
                    "successful_interactions": 0,
                    "average_response_time": 0.0,
                    "memory_fusion_calls": 0,
                    "pattern_learning_events": 0
                }
                
        except Exception as e:
            self.logger.error(f"Memory clear failed: {e}")
        
        return results
    
    def _update_performance_stats(self, success: bool, response_time: float):
        """Update performance statistics"""
        with self._lock:
            self.performance_stats["total_interactions"] += 1
            if success:
                self.performance_stats["successful_interactions"] += 1
            
            # Update average response time
            total = self.performance_stats["total_interactions"]
            current_avg = self.performance_stats["average_response_time"]
            self.performance_stats["average_response_time"] = (
                (current_avg * (total - 1) + response_time) / total
            )
    
    def export_session_data(self,
                           session_id: Optional[str] = None,
                           format: str = "json",
                           include_analytics: bool = True) -> Any:
        """
        Export session data from both memory systems
        
        Args:
            session_id: Session to export (current if None)
            format: Export format ("json" or "text")
            include_analytics: Include analytics data
            
        Returns:
            Exported session data
        """
        if not session_id:
            session_id = self.current_session_id
        
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "conversation_data": None,
            "context_data": None,
            "analytics": None
        }
        
        try:
            # Export from conversation memory
            if self.conversation_memory:
                export_data["conversation_data"] = self.conversation_memory.export_voice_session(
                    session_id=session_id,
                    format="json"  # Always get JSON from conversation memory
                )
            
            # Export from context memory (custom implementation)
            if self.context_memory:
                export_data["context_data"] = {
                    "conversation_history": [
                        conv for conv in self.context_memory.conversation_history
                        if conv.get("session_id") == session_id
                    ] if hasattr(self.context_memory, 'conversation_history') else [],
                    "memory_stats": self.context_memory.get_memory_stats()
                }
            
            # Include analytics if requested
            if include_analytics:
                export_data["analytics"] = self.get_comprehensive_analytics(days=1)
            
            if format == "text":
                # Convert to text format
                lines = [
                    f"Voice Memory Session Export: {session_id}",
                    f"Exported: {export_data['exported_at']}",
                    "=" * 50
                ]
                
                if export_data["conversation_data"]:
                    lines.append("\nConversation Data:")
                    conv_data = export_data["conversation_data"]
                    if isinstance(conv_data, dict) and "conversations" in conv_data:
                        for conv in conv_data["conversations"]:
                            lines.extend([
                                f"[{conv.get('timestamp', 'Unknown')}]",
                                f"User: {conv.get('user_input', 'N/A')}",
                                f"Assistant: {conv.get('assistant_response', 'N/A')}",
                                f"Confidence: {conv.get('transcription_confidence', 'N/A')}",
                                ""
                            ])
                
                return "\n".join(lines)
            
            return export_data
            
        except Exception as e:
            self.logger.error(f"Session export failed: {e}")
            export_data["error"] = str(e)
            return export_data
    
    def close(self):
        """Close memory manager and cleanup resources"""
        if self._closed:
            return
            
        try:
            # Save all memory before closing
            self.save_all_memory()
            
            # Close conversation memory
            if self.conversation_memory:
                self.conversation_memory.close()
                self.conversation_memory = None
            
            # Close thread executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            self._closed = True
            self.logger.info("🧠🎤 Voice Memory Manager closed successfully")
            
        except Exception as e:
            self.logger.error(f"Memory manager cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if not getattr(self, '_closed', True):
                self.close()
        except Exception:
            pass


# Convenience function for easy integration
def create_voice_memory_manager(config: Optional[Dict[str, Any]] = None) -> VoiceMemoryManager:
    """
    Create and initialize a Voice Memory Manager instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized VoiceMemoryManager instance
    """
    return VoiceMemoryManager(config)


if __name__ == "__main__":
    # Demo the Voice Memory Manager
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧠🎤 Voice Memory Manager Demo")
    print("=" * 50)
    
    # Create memory manager with demo config
    demo_config = {
        'enable_context_memory': True,
        'enable_audio_metadata': True,
        'conversation_db_path': 'demo_voice_conversations.db',
        'enable_pattern_learning': True,
        'enable_async_processing': True
    }
    
    manager = create_voice_memory_manager(demo_config)
    
    # Demo voice interaction
    audio_metadata = {
        'audio_file_path': '/tmp/demo_audio.wav',
        'confidence_score': 0.89,
        'audio_duration': 3.2,
        'processing_time': 1.5,
        'wake_word_detected': True,
        'wake_word_confidence': 0.92,
        'language': 'en-US'
    }
    
    processing_times = {
        'wake_word_time': 0.1,
        'transcription_time': 0.8,
        'llm_time': 0.5,
        'total_time': 1.4
    }
    
    audio_quality = {
        'noise_level': 0.3,
        'snr': 15.2,
        'clarity': 0.85,
        'noise_type': 'fan_noise'
    }
    
    # Add voice interaction
    result = manager.add_voice_interaction(
        user_input="What's the weather like today?",
        assistant_response="I'll check the current weather conditions for you.",
        audio_metadata=audio_metadata,
        agent_used="weather_agent",
        command_type="question",
        success=True,
        processing_times=processing_times,
        audio_quality=audio_quality
    )
    
    print(f"\n✅ Voice interaction added: {result}")
    
    # Add wake word event
    wake_result = manager.add_wake_word_event(
        wake_word="hey assistant",
        confidence=0.92,
        conversation_id=result.get('conversation_id'),
        detection_context={'detection_latency': 0.1}
    )
    
    print(f"\n🎯 Wake word event: {wake_result}")
    
    # Get fused context
    fused_context = manager.get_fused_context(
        context_window_size=5,
        include_audio_metadata=True
    )
    
    print(f"\n🔗 Fused context summary:")
    print(f"   - Conversations: {len(fused_context.get('conversation_history', []))}")
    print(f"   - Session ID: {fused_context.get('session_id')}")
    print(f"   - Memory fusion calls: {fused_context.get('memory_stats', {}).get('memory_fusion_calls', 0)}")
    
    # Get contextual hints
    hints = manager.get_contextual_response_hints("What's the temperature outside?")
    print(f"\n💡 Contextual hints: {len(hints.get('similar_past_interactions', []))} similar interactions found")
    
    # Search interactions
    search_results = manager.search_voice_interactions(
        query="weather",
        limit=5,
        filters={'min_confidence': 0.8}
    )
    
    print(f"\n🔍 Search results: {search_results.get('total_results', 0)} matches")
    
    # Get comprehensive analytics
    analytics = manager.get_comprehensive_analytics(days=1)
    print(f"\n📊 Analytics summary:")
    print(f"   - Total interactions: {analytics.get('system_performance', {}).get('total_interactions', 0)}")
    print(f"   - Success rate: {analytics.get('system_performance', {}).get('successful_interactions', 0)}")
    print(f"   - Recommendations: {len(analytics.get('recommendations', []))}")
    
    # Export session data
    export_data = manager.export_session_data(format="json", include_analytics=True)
    print(f"\n📤 Session export: {len(export_data.get('conversation_data', {}).get('conversations', []))} conversations")
    
    # Save and close
    save_results = manager.save_all_memory()
    print(f"\n💾 Memory saved: {save_results}")
    
    manager.close()
    print("\n✅ Voice Memory Manager Demo complete!")