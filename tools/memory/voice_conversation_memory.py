#!/usr/bin/env python3
"""
Voice Conversation Memory System using SQLite
Provides persistent storage and retrieval of voice conversation history
with specialized voice-specific features and analytics.
"""

import sqlite3
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import uuid


class VoiceConversationMemory:
    """SQLite-based voice conversation memory with voice-specific features"""
    
    def __init__(self, db_path: str = "data/voice_conversations.db", cache_size: int = 100):
        """
        Initialize voice conversation memory
        
        Args:
            db_path: Path to SQLite database file
            cache_size: Number of recent conversations to keep in memory
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup database
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Current session tracking - initialized before DB connection
        self.current_session_id = None
        self.session_start = datetime.now()
        
        # Connect with row factory for dict-like access
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Allow multi-threaded access
            isolation_level=None  # Autocommit mode
        )
        self.conn.row_factory = sqlite3.Row
        
        # Initialize database schema
        self._init_database()
        
        # Determine session ID after database is ready
        self._determine_session_id()
        
        # In-memory cache for recent conversations
        self.cache = deque(maxlen=cache_size)
        self.cache_size = cache_size
        
        # Load recent conversations into cache
        self._load_cache()
        
        self.logger.info(f"✅ Voice conversation memory initialized at {self.db_path}")
    
    def _init_database(self):
        """Create database tables and indexes with voice-specific schema"""
        cursor = self.conn.cursor()
        
        # Main voice conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                agent_used TEXT,
                command_type TEXT,
                success BOOLEAN DEFAULT 1,
                response_time REAL,
                metadata JSON,
                
                -- Voice-specific fields
                audio_file_path TEXT,
                audio_duration REAL,
                transcription_confidence REAL,
                wake_word_detected BOOLEAN DEFAULT 0,
                wake_word_confidence REAL,
                noise_level REAL,
                voice_activity_ratio REAL,
                
                -- Processing time breakdown
                wake_word_detection_time REAL,
                transcription_time REAL,
                llm_processing_time REAL,
                tts_generation_time REAL,
                total_processing_time REAL,
                
                -- Audio quality metrics
                signal_to_noise_ratio REAL,
                audio_clarity_score REAL,
                background_noise_type TEXT,
                voice_pitch_range TEXT,
                speaking_rate REAL,
                
                -- Embedding for similarity search
                embedding BLOB
            )
        """)
        
        # Voice sessions table with voice-specific metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_sessions (
                id TEXT PRIMARY KEY,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                total_messages INTEGER DEFAULT 0,
                successful_commands INTEGER DEFAULT 0,
                failed_commands INTEGER DEFAULT 0,
                agents_used JSON,
                user_metadata JSON,
                
                -- Voice session metrics
                total_audio_duration REAL DEFAULT 0,
                average_confidence REAL DEFAULT 0,
                wake_word_accuracy REAL DEFAULT 0,
                total_wake_words INTEGER DEFAULT 0,
                successful_wake_words INTEGER DEFAULT 0,
                average_noise_level REAL DEFAULT 0,
                primary_background_noise TEXT,
                session_quality_score REAL DEFAULT 0,
                user_satisfaction_score REAL,
                
                -- Performance metrics
                average_response_time REAL DEFAULT 0,
                fastest_response_time REAL,
                slowest_response_time REAL,
                total_processing_time REAL DEFAULT 0
            )
        """)
        
        # Audio files tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                file_size INTEGER,
                duration REAL,
                sample_rate INTEGER,
                channels INTEGER,
                bit_depth INTEGER,
                format TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                conversation_id INTEGER,
                
                -- Audio analysis results
                noise_profile JSON,
                voice_segments JSON,
                audio_features JSON,
                quality_assessment JSON,
                
                FOREIGN KEY (conversation_id) REFERENCES voice_conversations(id)
            )
        """)
        
        # Wake word detection events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wake_word_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                conversation_id INTEGER,
                wake_word TEXT,
                confidence REAL,
                false_positive BOOLEAN DEFAULT 0,
                audio_context_before REAL,
                audio_context_after REAL,
                detection_latency REAL,
                background_context TEXT,
                
                FOREIGN KEY (session_id) REFERENCES voice_sessions(id),
                FOREIGN KEY (conversation_id) REFERENCES voice_conversations(id)
            )
        """)
        
        # Transcription quality events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcription_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL,
                word_count INTEGER,
                uncertain_words INTEGER,
                correction_needed BOOLEAN DEFAULT 0,
                user_correction TEXT,
                quality_issues JSON,
                language_detected TEXT,
                accent_detected TEXT,
                
                FOREIGN KEY (conversation_id) REFERENCES voice_conversations(id)
            )
        """)
        
        # Voice command patterns for learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_text TEXT NOT NULL,
                normalized_pattern TEXT,
                frequency INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 1.0,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_intent TEXT,
                context_tags JSON,
                pattern_variations JSON
            )
        """)
        
        # User voice preferences and settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                preference_type TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                learned_from_interactions BOOLEAN DEFAULT 1,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata JSON
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_voice_conv_timestamp 
            ON voice_conversations(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_voice_conv_session 
            ON voice_conversations(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_voice_conv_confidence 
            ON voice_conversations(transcription_confidence DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_voice_conv_wake_word 
            ON voice_conversations(wake_word_detected, wake_word_confidence)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audio_files_hash 
            ON audio_files(file_hash)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_wake_word_events_session 
            ON wake_word_events(session_id, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcription_quality_conv 
            ON transcription_quality(conversation_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_voice_patterns_type 
            ON voice_patterns(pattern_type, frequency DESC)
        """)
        
        # Create full-text search virtual table for voice conversations
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS voice_conversations_fts
            USING fts5(
                user_input, 
                assistant_response,
                content=voice_conversations,
                content_rowid=id
            )
        """)
        
        # Trigger to keep FTS index updated
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS voice_conversations_ai 
            AFTER INSERT ON voice_conversations 
            BEGIN
                INSERT INTO voice_conversations_fts(rowid, user_input, assistant_response)
                VALUES (new.id, new.user_input, new.assistant_response);
            END
        """)
        
        self.conn.commit()
        
        # Start new session
        self._start_session()
    
    def _determine_session_id(self):
        """Determine whether to continue recent session or create new one"""
        try:
            cursor = self.conn.cursor()
            
            # Check for recent sessions (within last 4 hours)
            cutoff_time = datetime.now() - timedelta(hours=4)
            cursor.execute("""
                SELECT id, start_time, end_time FROM voice_sessions
                WHERE start_time > ? AND (end_time IS NULL OR end_time > ?)
                ORDER BY start_time DESC LIMIT 1
            """, (cutoff_time.isoformat(), cutoff_time.isoformat()))
            
            recent_session = cursor.fetchone()
            
            if recent_session and not recent_session['end_time']:
                # Continue the recent unfinished session
                self.current_session_id = recent_session['id']
                self.session_start = datetime.fromisoformat(recent_session['start_time'])
                self.logger.info(f"Continuing recent session: {self.current_session_id}")
            else:
                # Create new session
                self.current_session_id = str(uuid.uuid4())
                self.logger.info(f"Starting new session: {self.current_session_id}")
                
        except Exception as e:
            # Fallback to new session if anything goes wrong
            self.current_session_id = str(uuid.uuid4())
            self.logger.warning(f"Could not determine session, creating new: {e}")
    
    def _start_session(self):
        """Start a new voice conversation session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO voice_sessions (id, start_time)
            VALUES (?, ?)
        """, (self.current_session_id, self.session_start))
        self.conn.commit()
    
    def _load_cache(self):
        """Load recent conversations into memory cache"""
        recent = self.get_recent(self.cache_size)
        self.cache.clear()
        self.cache.extend(recent)
    
    def add_voice_conversation(self, 
                              user_input: str, 
                              assistant_response: str,
                              agent_used: str = None,
                              command_type: str = None,
                              success: bool = True,
                              response_time: float = None,
                              metadata: Dict = None,
                              # Voice-specific parameters
                              audio_file_path: str = None,
                              audio_duration: float = None,
                              transcription_confidence: float = None,
                              wake_word_detected: bool = False,
                              wake_word_confidence: float = None,
                              noise_level: float = None,
                              voice_activity_ratio: float = None,
                              # Processing times
                              wake_word_detection_time: float = None,
                              transcription_time: float = None,
                              llm_processing_time: float = None,
                              tts_generation_time: float = None,
                              total_processing_time: float = None,
                              # Audio quality
                              signal_to_noise_ratio: float = None,
                              audio_clarity_score: float = None,
                              background_noise_type: str = None,
                              voice_pitch_range: str = None,
                              speaking_rate: float = None) -> int:
        """
        Add a voice conversation to memory with voice-specific data
        
        Returns:
            Conversation ID
        """
        cursor = self.conn.cursor()
        
        # Add to database
        cursor.execute("""
            INSERT INTO voice_conversations 
            (session_id, user_input, assistant_response, agent_used, 
             command_type, success, response_time, metadata,
             audio_file_path, audio_duration, transcription_confidence,
             wake_word_detected, wake_word_confidence, noise_level,
             voice_activity_ratio, wake_word_detection_time, transcription_time,
             llm_processing_time, tts_generation_time, total_processing_time,
             signal_to_noise_ratio, audio_clarity_score, background_noise_type,
             voice_pitch_range, speaking_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.current_session_id,
            user_input,
            assistant_response,
            agent_used,
            command_type,
            success,
            response_time,
            json.dumps(metadata) if metadata else None,
            audio_file_path,
            audio_duration,
            transcription_confidence,
            wake_word_detected,
            wake_word_confidence,
            noise_level,
            voice_activity_ratio,
            wake_word_detection_time,
            transcription_time,
            llm_processing_time,
            tts_generation_time,
            total_processing_time,
            signal_to_noise_ratio,
            audio_clarity_score,
            background_noise_type,
            voice_pitch_range,
            speaking_rate
        ))
        
        conversation_id = cursor.lastrowid
        
        # Update session statistics
        cursor.execute("""
            UPDATE voice_sessions 
            SET total_messages = total_messages + 1,
                successful_commands = successful_commands + ?,
                failed_commands = failed_commands + ?,
                total_audio_duration = total_audio_duration + ?,
                total_processing_time = total_processing_time + ?
            WHERE id = ?
        """, (
            1 if success else 0, 
            0 if success else 1, 
            audio_duration or 0,
            total_processing_time or 0,
            self.current_session_id
        ))
        
        # Update wake word statistics if detected
        if wake_word_detected:
            cursor.execute("""
                UPDATE voice_sessions 
                SET total_wake_words = total_wake_words + 1,
                    successful_wake_words = successful_wake_words + 1
                WHERE id = ?
            """, (self.current_session_id,))
        
        # Add to cache
        entry = {
            "id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "user": user_input,
            "assistant": assistant_response,
            "agent": agent_used,
            "command_type": command_type,
            "success": success,
            "response_time": response_time,
            "metadata": metadata,
            "transcription_confidence": transcription_confidence,
            "wake_word_detected": wake_word_detected,
            "audio_duration": audio_duration
        }
        self.cache.append(entry)
        
        self.logger.debug(f"Added voice conversation {conversation_id} to memory")
        return conversation_id
    
    def add_audio_file(self, 
                       file_path: str,
                       conversation_id: int = None,
                       audio_analysis: Dict = None) -> int:
        """
        Register an audio file in the database
        
        Args:
            file_path: Path to the audio file
            conversation_id: Associated conversation ID
            audio_analysis: Results from audio analysis
            
        Returns:
            Audio file ID
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Calculate file hash for deduplication
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        file_size = Path(file_path).stat().st_size
        
        cursor = self.conn.cursor()
        
        # Extract audio metadata if analysis provided
        duration = audio_analysis.get('duration') if audio_analysis else None
        sample_rate = audio_analysis.get('sample_rate') if audio_analysis else None
        channels = audio_analysis.get('channels') if audio_analysis else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO audio_files 
            (file_path, file_hash, file_size, duration, sample_rate, channels,
             conversation_id, noise_profile, voice_segments, audio_features, quality_assessment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(file_path),
            file_hash,
            file_size,
            duration,
            sample_rate,
            channels,
            conversation_id,
            json.dumps(audio_analysis.get('noise_profile')) if audio_analysis else None,
            json.dumps(audio_analysis.get('voice_segments')) if audio_analysis else None,
            json.dumps(audio_analysis.get('features')) if audio_analysis else None,
            json.dumps(audio_analysis.get('quality')) if audio_analysis else None
        ))
        
        return cursor.lastrowid
    
    def add_wake_word_event(self,
                           wake_word: str,
                           confidence: float,
                           conversation_id: int = None,
                           false_positive: bool = False,
                           detection_latency: float = None,
                           background_context: str = None) -> int:
        """
        Record a wake word detection event
        
        Returns:
            Wake word event ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO wake_word_events 
            (session_id, conversation_id, wake_word, confidence, false_positive,
             detection_latency, background_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self.current_session_id,
            conversation_id,
            wake_word,
            confidence,
            false_positive,
            detection_latency,
            background_context
        ))
        
        return cursor.lastrowid
    
    def add_transcription_quality(self,
                                 conversation_id: int,
                                 confidence_score: float,
                                 word_count: int,
                                 uncertain_words: int,
                                 correction_needed: bool = False,
                                 user_correction: str = None,
                                 quality_issues: Dict = None,
                                 language_detected: str = None) -> int:
        """
        Record transcription quality metrics
        
        Returns:
            Quality record ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO transcription_quality 
            (conversation_id, confidence_score, word_count, uncertain_words,
             correction_needed, user_correction, quality_issues, language_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id,
            confidence_score,
            word_count,
            uncertain_words,
            correction_needed,
            user_correction,
            json.dumps(quality_issues) if quality_issues else None,
            language_detected
        ))
        
        return cursor.lastrowid
    
    def learn_voice_pattern(self,
                           pattern_type: str,
                           pattern_text: str,
                           user_intent: str = None,
                           context_tags: List[str] = None) -> int:
        """
        Learn and store a voice command pattern
        
        Args:
            pattern_type: Type of pattern (command, question, etc.)
            pattern_text: The actual text pattern
            user_intent: What the user intended
            context_tags: Context tags for this pattern
            
        Returns:
            Pattern ID
        """
        cursor = self.conn.cursor()
        
        # Normalize pattern for matching
        normalized = pattern_text.lower().strip()
        
        # Check if pattern already exists
        cursor.execute("""
            SELECT id, frequency FROM voice_patterns 
            WHERE pattern_type = ? AND normalized_pattern = ?
        """, (pattern_type, normalized))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update frequency
            cursor.execute("""
                UPDATE voice_patterns 
                SET frequency = frequency + 1, last_used = ?
                WHERE id = ?
            """, (datetime.now(), existing['id']))
            return existing['id']
        else:
            # Insert new pattern
            cursor.execute("""
                INSERT INTO voice_patterns 
                (pattern_type, pattern_text, normalized_pattern, user_intent, context_tags)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pattern_type,
                pattern_text,
                normalized,
                user_intent,
                json.dumps(context_tags) if context_tags else None
            ))
            return cursor.lastrowid
    
    def get_recent(self, limit: int = 10, session_only: bool = False) -> List[Dict]:
        """Get recent voice conversations with voice-specific data"""
        if not session_only and len(self.cache) >= limit:
            return list(self.cache)[-limit:]
        
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM voice_conversations
            {} ORDER BY timestamp DESC LIMIT ?
        """.format("WHERE session_id = ?" if session_only else "")
        
        params = [self.current_session_id, limit] if session_only else [limit]
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        
        conversations = []
        for row in reversed(rows):
            conversations.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "session_id": row["session_id"],
                "user": row["user_input"],
                "assistant": row["assistant_response"],
                "agent": row["agent_used"],
                "command_type": row["command_type"],
                "success": bool(row["success"]),
                "response_time": row["response_time"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                "audio_file_path": row["audio_file_path"],
                "audio_duration": row["audio_duration"],
                "transcription_confidence": row["transcription_confidence"],
                "wake_word_detected": bool(row["wake_word_detected"]),
                "noise_level": row["noise_level"],
                "total_processing_time": row["total_processing_time"]
            })
        
        return conversations
    
    def search_voice_conversations(self, 
                                  query: str, 
                                  limit: int = 10,
                                  min_confidence: float = None,
                                  wake_word_only: bool = False) -> List[Dict]:
        """
        Search voice conversations with voice-specific filters
        
        Args:
            query: Search query
            limit: Maximum results
            min_confidence: Minimum transcription confidence
            wake_word_only: Only include wake word triggered conversations
        """
        cursor = self.conn.cursor()
        
        base_query = """
            SELECT c.* 
            FROM voice_conversations c
            JOIN voice_conversations_fts fts ON c.id = fts.rowid
            WHERE voice_conversations_fts MATCH ?
        """
        
        conditions = []
        params = [query]
        
        if min_confidence is not None:
            conditions.append("c.transcription_confidence >= ?")
            params.append(min_confidence)
        
        if wake_word_only:
            conditions.append("c.wake_word_detected = 1")
        
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        base_query += " ORDER BY rank LIMIT ?"
        params.append(limit)
        
        cursor.execute(base_query, params)
        
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "user": row["user_input"],
                "assistant": row["assistant_response"],
                "agent": row["agent_used"],
                "transcription_confidence": row["transcription_confidence"],
                "wake_word_detected": bool(row["wake_word_detected"]),
                "score": 1.0  # FTS rank
            })
        
        return results
    
    def get_context_window(self, size: int = 5, include_metadata: bool = False) -> str:
        """
        Get formatted context window for LLM
        
        Args:
            size: Number of recent conversations to include
            include_metadata: Whether to include metadata in context
            
        Returns:
            Formatted context string
        """
        recent = self.get_recent(size)
        
        if not recent:
            return "No previous conversation history."
        
        context_parts = []
        
        for conv in recent:
            user_msg = f"User: {conv['user']}"
            assistant_msg = f"Assistant: {conv['assistant']}"
            
            if include_metadata and conv.get('agent'):
                assistant_msg += f" [via {conv['agent']}]"
            
            context_parts.append(user_msg)
            context_parts.append(assistant_msg)
        
        return "\n".join(context_parts)
    
    def get_voice_analytics(self, days: int = 7) -> Dict:
        """
        Get comprehensive voice interaction analytics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analytics dictionary with voice metrics
        """
        cursor = self.conn.cursor()
        since_date = datetime.now() - timedelta(days=days)
        
        # Basic conversation stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_conversations,
                AVG(transcription_confidence) as avg_confidence,
                COUNT(CASE WHEN wake_word_detected = 1 THEN 1 END) as wake_word_conversations,
                AVG(audio_duration) as avg_audio_duration,
                AVG(total_processing_time) as avg_processing_time,
                AVG(noise_level) as avg_noise_level,
                COUNT(CASE WHEN transcription_confidence < 0.7 THEN 1 END) as low_confidence_count
            FROM voice_conversations
            WHERE timestamp > ?
        """, (since_date,))
        
        basic_stats = cursor.fetchone()
        
        # Processing time breakdown
        cursor.execute("""
            SELECT 
                AVG(wake_word_detection_time) as avg_wake_word_time,
                AVG(transcription_time) as avg_transcription_time,
                AVG(llm_processing_time) as avg_llm_time,
                AVG(tts_generation_time) as avg_tts_time
            FROM voice_conversations
            WHERE timestamp > ? AND total_processing_time IS NOT NULL
        """, (since_date,))
        
        processing_stats = cursor.fetchone()
        
        # Wake word performance
        cursor.execute("""
            SELECT 
                COUNT(*) as total_wake_events,
                AVG(confidence) as avg_wake_confidence,
                COUNT(CASE WHEN false_positive = 1 THEN 1 END) as false_positives,
                AVG(detection_latency) as avg_detection_latency
            FROM wake_word_events
            WHERE timestamp > ?
        """, (since_date,))
        
        wake_word_stats = cursor.fetchone()
        
        # Audio quality trends
        cursor.execute("""
            SELECT 
                AVG(signal_to_noise_ratio) as avg_snr,
                AVG(audio_clarity_score) as avg_clarity,
                COUNT(DISTINCT background_noise_type) as noise_variety,
                background_noise_type,
                COUNT(*) as noise_count
            FROM voice_conversations
            WHERE timestamp > ? AND background_noise_type IS NOT NULL
            GROUP BY background_noise_type
            ORDER BY noise_count DESC
        """, (since_date,))
        
        noise_analysis = cursor.fetchall()
        
        # Most common voice patterns
        cursor.execute("""
            SELECT pattern_type, pattern_text, frequency, success_rate
            FROM voice_patterns
            ORDER BY frequency DESC
            LIMIT 10
        """)
        
        common_patterns = cursor.fetchall()
        
        return {
            "analysis_period_days": days,
            "basic_stats": {
                "total_conversations": basic_stats["total_conversations"],
                "average_confidence": round(basic_stats["avg_confidence"] or 0, 3),
                "wake_word_conversations": basic_stats["wake_word_conversations"],
                "wake_word_rate": round((basic_stats["wake_word_conversations"] or 0) / 
                                      max(basic_stats["total_conversations"], 1), 3),
                "average_audio_duration": round(basic_stats["avg_audio_duration"] or 0, 2),
                "average_processing_time": round(basic_stats["avg_processing_time"] or 0, 3),
                "average_noise_level": round(basic_stats["avg_noise_level"] or 0, 3),
                "low_confidence_rate": round((basic_stats["low_confidence_count"] or 0) / 
                                           max(basic_stats["total_conversations"], 1), 3)
            },
            "processing_performance": {
                "avg_wake_word_time": round(processing_stats["avg_wake_word_time"] or 0, 3),
                "avg_transcription_time": round(processing_stats["avg_transcription_time"] or 0, 3),
                "avg_llm_time": round(processing_stats["avg_llm_time"] or 0, 3),
                "avg_tts_time": round(processing_stats["avg_tts_time"] or 0, 3)
            },
            "wake_word_performance": {
                "total_events": wake_word_stats["total_wake_events"],
                "average_confidence": round(wake_word_stats["avg_wake_confidence"] or 0, 3),
                "false_positive_count": wake_word_stats["false_positives"],
                "false_positive_rate": round((wake_word_stats["false_positives"] or 0) / 
                                           max(wake_word_stats["total_wake_events"], 1), 3),
                "average_detection_latency": round(wake_word_stats["avg_detection_latency"] or 0, 3)
            },
            "audio_quality": {
                "noise_environments": [dict(row) for row in noise_analysis],
                "primary_noise_type": noise_analysis[0]["background_noise_type"] if noise_analysis else None
            },
            "common_patterns": [dict(row) for row in common_patterns]
        }
    
    def get_transcription_quality_report(self, days: int = 7) -> Dict:
        """Generate transcription quality report"""
        cursor = self.conn.cursor()
        since_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT 
                AVG(confidence_score) as avg_confidence,
                AVG(word_count) as avg_word_count,
                AVG(uncertain_words) as avg_uncertain_words,
                COUNT(CASE WHEN correction_needed = 1 THEN 1 END) as corrections_needed,
                COUNT(*) as total_transcriptions,
                language_detected,
                COUNT(*) as lang_count
            FROM transcription_quality tq
            JOIN voice_conversations vc ON tq.conversation_id = vc.id
            WHERE vc.timestamp > ?
            GROUP BY language_detected
        """, (since_date,))
        
        quality_data = cursor.fetchall()
        
        return {
            "period_days": days,
            "quality_metrics": [dict(row) for row in quality_data],
            "summary": {
                "total_transcriptions": sum(row["total_transcriptions"] for row in quality_data),
                "overall_avg_confidence": round(
                    sum(row["avg_confidence"] * row["total_transcriptions"] for row in quality_data) /
                    max(sum(row["total_transcriptions"] for row in quality_data), 1), 3
                ),
                "correction_rate": round(
                    sum(row["corrections_needed"] for row in quality_data) /
                    max(sum(row["total_transcriptions"] for row in quality_data), 1), 3
                )
            }
        }
    
    def optimize_voice_patterns(self, min_frequency: int = 3) -> List[Dict]:
        """
        Get optimization suggestions based on voice patterns
        
        Args:
            min_frequency: Minimum frequency to consider for optimization
            
        Returns:
            List of optimization suggestions
        """
        cursor = self.conn.cursor()
        
        # Find patterns with low success rates
        cursor.execute("""
            SELECT * FROM voice_patterns
            WHERE frequency >= ? AND success_rate < 0.8
            ORDER BY frequency DESC, success_rate ASC
        """, (min_frequency,))
        
        problematic_patterns = cursor.fetchall()
        
        suggestions = []
        for pattern in problematic_patterns:
            suggestions.append({
                "type": "improve_pattern",
                "pattern": pattern["pattern_text"],
                "frequency": pattern["frequency"],
                "success_rate": pattern["success_rate"],
                "suggestion": f"Pattern '{pattern['pattern_text']}' has low success rate ({pattern['success_rate']:.2f}). Consider alternative phrasing or clarification."
            })
        
        return suggestions
    
    def export_voice_session(self, session_id: str = None, format: str = "json") -> Any:
        """Export voice session data including audio metrics"""
        if not session_id:
            session_id = self.current_session_id
        
        cursor = self.conn.cursor()
        
        # Get session info
        cursor.execute("SELECT * FROM voice_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        # Get conversations with voice data
        cursor.execute("""
            SELECT * FROM voice_conversations 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_id,))
        conversations = cursor.fetchall()
        
        # Get wake word events
        cursor.execute("""
            SELECT * FROM wake_word_events 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_id,))
        wake_events = cursor.fetchall()
        
        if format == "json":
            return {
                "session": dict(session) if session else None,
                "conversations": [dict(c) for c in conversations],
                "wake_word_events": [dict(w) for w in wake_events],
                "exported_at": datetime.now().isoformat(),
                "voice_analytics": self.get_voice_analytics(days=1) if conversations else {}
            }
        
        elif format == "text":
            lines = [
                f"Voice Session: {session_id}",
                f"Exported: {datetime.now()}",
                f"Total Conversations: {len(conversations)}",
                f"Wake Word Events: {len(wake_events)}",
                ""
            ]
            
            for conv in conversations:
                lines.extend([
                    f"[{conv['timestamp']}]",
                    f"User: {conv['user_input']}",
                    f"Assistant: {conv['assistant_response']}",
                    f"Confidence: {conv['transcription_confidence']:.3f}" if conv['transcription_confidence'] else "",
                    f"Duration: {conv['audio_duration']:.2f}s" if conv['audio_duration'] else "",
                    ""
                ])
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def close(self):
        """Close database connection and end session"""
        if self.conn:
            # Update session end time
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE voice_sessions 
                SET end_time = ?
                WHERE id = ?
            """, (datetime.now(), self.current_session_id))
            
            self.conn.close()
            self.logger.info("Voice conversation memory closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.close()
        except Exception:
            pass


# Convenience function for easy integration
def create_voice_memory(db_path: str = None) -> VoiceConversationMemory:
    """
    Create and initialize a voice conversation memory instance
    
    Args:
        db_path: Optional custom database path
        
    Returns:
        Initialized VoiceConversationMemory instance
    """
    if db_path is None:
        db_path = "data/voice_conversations.db"
    
    return VoiceConversationMemory(db_path)


if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create memory instance
    memory = create_voice_memory("test_voice_conversations.db")
    
    # Add a sample conversation
    conv_id = memory.add_voice_conversation(
        user_input="What's the weather like?",
        assistant_response="I'll check the weather for you.",
        transcription_confidence=0.95,
        wake_word_detected=True,
        wake_word_confidence=0.87,
        audio_duration=2.3,
        noise_level=0.2,
        transcription_time=0.5,
        llm_processing_time=1.2,
        total_processing_time=1.8
    )
    
    # Add transcription quality data
    memory.add_transcription_quality(
        conversation_id=conv_id,
        confidence_score=0.95,
        word_count=6,
        uncertain_words=0,
        language_detected="en-US"
    )
    
    # Get analytics
    analytics = memory.get_voice_analytics(days=1)
    print("Voice Analytics:")
    print(json.dumps(analytics, indent=2))
    
    # Search conversations
    results = memory.search_voice_conversations("weather", min_confidence=0.9)
    print(f"\nFound {len(results)} high-confidence conversations about weather")
    
    memory.close()
    print("Test completed successfully!")