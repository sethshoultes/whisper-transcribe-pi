#!/usr/bin/env python3
"""
Voice Memory Setup and Migration Script
Handles database initialization, schema updates, and data migration
"""

import sqlite3
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class VoiceMemorySetup:
    """Setup and migration utilities for voice conversation memory"""
    
    def __init__(self, db_path: str = "data/voice_conversations.db"):
        """
        Initialize setup manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Schema version tracking
        self.current_version = 1
        self.migrations = {
            1: self._migrate_to_v1
        }
    
    def get_schema_version(self) -> int:
        """Get current database schema version"""
        if not self.db_path.exists():
            return 0
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if schema_version table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return 0
            
            # Get version
            cursor.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            
            return row[0] if row else 0
            
        except Exception as e:
            self.logger.error(f"Error checking schema version: {e}")
            return 0
    
    def create_fresh_database(self):
        """Create a fresh database with the latest schema"""
        if self.db_path.exists():
            backup_path = self.db_path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Created backup at {backup_path}")
        
        # Import and create fresh instance
        from voice_conversation_memory import VoiceConversationMemory
        
        memory = VoiceConversationMemory(str(self.db_path))
        
        # Set schema version
        self._set_schema_version(memory.conn, self.current_version)
        memory.close()
        
        self.logger.info(f"Created fresh database at {self.db_path}")
    
    def migrate_database(self, target_version: int = None):
        """
        Migrate database to target version
        
        Args:
            target_version: Target schema version (None for latest)
        """
        if target_version is None:
            target_version = self.current_version
        
        current_version = self.get_schema_version()
        
        if current_version == target_version:
            self.logger.info(f"Database already at version {target_version}")
            return
        
        if current_version > target_version:
            self.logger.error(f"Cannot downgrade from v{current_version} to v{target_version}")
            return
        
        self.logger.info(f"Migrating database from v{current_version} to v{target_version}")
        
        # Create backup before migration
        if self.db_path.exists():
            backup_path = self.db_path.with_suffix(f'.backup.v{current_version}.db')
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Created migration backup at {backup_path}")
        
        # Run migrations
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        try:
            for version in range(current_version + 1, target_version + 1):
                if version in self.migrations:
                    self.logger.info(f"Running migration to v{version}")
                    self.migrations[version](conn)
                    self._set_schema_version(conn, version)
                    conn.commit()
                else:
                    self.logger.error(f"No migration found for version {version}")
                    conn.rollback()
                    return
            
            self.logger.info(f"Migration completed successfully to v{target_version}")
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _set_schema_version(self, conn: sqlite3.Connection, version: int):
        """Set schema version in database"""
        cursor = conn.cursor()
        
        # Create schema_version table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                migrated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)
        
        # Insert version record
        cursor.execute("""
            INSERT INTO schema_version (version, notes)
            VALUES (?, ?)
        """, (version, f"Schema version {version}"))
    
    def _migrate_to_v1(self, conn: sqlite3.Connection):
        """Migration to version 1 (initial voice memory schema)"""
        cursor = conn.cursor()
        
        # This would contain any specific migration logic
        # For now, it's a placeholder since v1 is the initial schema
        self.logger.info("Version 1 migration completed (initial schema)")
    
    def validate_database(self) -> Dict:
        """
        Validate database integrity and structure
        
        Returns:
            Validation report dictionary
        """
        if not self.db_path.exists():
            return {"valid": False, "error": "Database file does not exist"}
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check required tables
            required_tables = [
                'voice_conversations',
                'voice_sessions',
                'audio_files',
                'wake_word_events',
                'transcription_quality',
                'voice_patterns',
                'voice_user_preferences',
                'voice_conversations_fts',
                'schema_version'
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [t for t in required_tables if t not in existing_tables]
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Basic data integrity checks
            integrity_issues = []
            
            # Check for orphaned records
            cursor.execute("""
                SELECT COUNT(*) FROM voice_conversations vc
                LEFT JOIN voice_sessions vs ON vc.session_id = vs.id
                WHERE vs.id IS NULL
            """)
            orphaned_conversations = cursor.fetchone()[0]
            
            if orphaned_conversations > 0:
                integrity_issues.append(f"{orphaned_conversations} conversations with missing sessions")
            
            # Check foreign key constraints
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            
            if fk_violations:
                integrity_issues.append(f"{len(fk_violations)} foreign key violations")
            
            conn.close()
            
            return {
                "valid": len(missing_tables) == 0 and len(integrity_issues) == 0,
                "schema_version": self.get_schema_version(),
                "missing_tables": missing_tables,
                "total_tables": len(existing_tables),
                "total_indexes": len(indexes),
                "integrity_issues": integrity_issues,
                "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def export_database_info(self) -> Dict:
        """Export comprehensive database information"""
        if not self.db_path.exists():
            return {"error": "Database does not exist"}
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get table statistics
            table_stats = {}
            
            # Voice conversations
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest,
                    AVG(transcription_confidence) as avg_confidence,
                    COUNT(CASE WHEN wake_word_detected = 1 THEN 1 END) as wake_word_count
                FROM voice_conversations
            """)
            row = cursor.fetchone()
            if row["total"] > 0:
                table_stats["voice_conversations"] = dict(row)
            
            # Voice sessions
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN end_time IS NOT NULL THEN 1 END) as completed,
                    AVG(total_messages) as avg_messages_per_session,
                    AVG(total_audio_duration) as avg_audio_duration
                FROM voice_sessions
            """)
            row = cursor.fetchone()
            if row["total"] > 0:
                table_stats["voice_sessions"] = dict(row)
            
            # Audio files
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(file_size) as total_size,
                    AVG(duration) as avg_duration
                FROM audio_files
            """)
            row = cursor.fetchone()
            if row["total"] > 0:
                table_stats["audio_files"] = dict(row)
            
            # Wake word events
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN false_positive = 1 THEN 1 END) as false_positives
                FROM wake_word_events
            """)
            row = cursor.fetchone()
            if row["total"] > 0:
                table_stats["wake_word_events"] = dict(row)
            
            # Voice patterns
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    pattern_type,
                    COUNT(*) as count,
                    AVG(frequency) as avg_frequency
                FROM voice_patterns
                GROUP BY pattern_type
            """)
            patterns_by_type = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "database_path": str(self.db_path),
                "schema_version": self.get_schema_version(),
                "file_size_bytes": self.db_path.stat().st_size,
                "table_statistics": table_stats,
                "voice_patterns_by_type": patterns_by_type,
                "exported_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_old_data(self, days: int = 30) -> Dict:
        """
        Clean up old data from the database
        
        Args:
            days: Keep data newer than this many days
            
        Returns:
            Cleanup report
        """
        if not self.db_path.exists():
            return {"error": "Database does not exist"}
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Count records to be deleted
            cursor.execute("""
                SELECT COUNT(*) FROM voice_conversations WHERE timestamp < ?
            """, (cutoff_date,))
            old_conversations = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM wake_word_events WHERE timestamp < ?
            """, (cutoff_date,))
            old_wake_events = cursor.fetchone()[0]
            
            # Delete old data
            cursor.execute("""
                DELETE FROM voice_conversations WHERE timestamp < ?
            """, (cutoff_date,))
            
            cursor.execute("""
                DELETE FROM wake_word_events WHERE timestamp < ?
            """, (cutoff_date,))
            
            # Clean up orphaned sessions
            cursor.execute("""
                DELETE FROM voice_sessions
                WHERE id NOT IN (
                    SELECT DISTINCT session_id FROM voice_conversations
                )
            """)
            orphaned_sessions = cursor.rowcount
            
            # Clean up orphaned audio files
            cursor.execute("""
                DELETE FROM audio_files
                WHERE conversation_id NOT IN (
                    SELECT id FROM voice_conversations
                )
            """)
            orphaned_audio = cursor.rowcount
            
            # Rebuild FTS index
            cursor.execute("INSERT INTO voice_conversations_fts(voice_conversations_fts) VALUES('rebuild')")
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "days_kept": days,
                "deleted_conversations": old_conversations,
                "deleted_wake_events": old_wake_events,
                "deleted_sessions": orphaned_sessions,
                "deleted_audio_files": orphaned_audio,
                "cleanup_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Command line interface for voice memory setup"""
    parser = argparse.ArgumentParser(description="Voice Memory Database Setup and Migration")
    
    parser.add_argument(
        "--db-path", 
        default="data/voice_conversations.db",
        help="Path to database file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize fresh database")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate database")
    migrate_parser.add_argument("--version", type=int, help="Target version")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate database")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show database information")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old data")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Days to keep")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    setup = VoiceMemorySetup(args.db_path)
    
    if args.command == "init":
        setup.create_fresh_database()
        print(f"✅ Fresh database created at {args.db_path}")
        
    elif args.command == "migrate":
        setup.migrate_database(args.version)
        print(f"✅ Migration completed")
        
    elif args.command == "validate":
        report = setup.validate_database()
        print(json.dumps(report, indent=2))
        
    elif args.command == "info":
        info = setup.export_database_info()
        print(json.dumps(info, indent=2))
        
    elif args.command == "cleanup":
        report = setup.cleanup_old_data(args.days)
        print(json.dumps(report, indent=2))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()