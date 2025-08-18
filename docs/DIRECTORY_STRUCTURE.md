# Directory Structure

## Overview
The Whisper Transcribe Pi project has been reorganized for better maintainability and clarity.

## Structure

```
whisper-transcribe-pi/
├── whisper_transcribe.py         # Standard version main file
├── whisper_transcribe_pro.py     # Pro version main file
├── requirements.txt               # Python dependencies
├── setup_pro.sh                   # Pro version setup script
├── install.sh                     # Standard version setup script
├── launch_whisper.sh              # Standard version launcher
├── launch_whisper_pro.sh          # Pro version launcher
├── README.md                      # Main documentation
├── CLAUDE.md                      # Claude Code development guide
├── LICENSE                        # License file
├── *.desktop                      # Desktop entry files
│
├── docs/                          # All documentation
│   ├── planning/                  # Planning documents
│   │   ├── MARKETING.md
│   │   ├── PROJECT_MAP.md
│   │   └── PRO_FEATURES.md
│   │
│   ├── implementation/            # Implementation summaries
│   │   ├── AI_PROVIDER_UPDATE_SUMMARY.md
│   │   ├── MEMORY_FIXES_SUMMARY.md
│   │   ├── MEMORY_SYSTEM_IMPLEMENTATION_PLAN.md
│   │   ├── OPENAI_FIX_SUMMARY.md
│   │   ├── VOICE_CONTROLLED_AI_INTEGRATION_PLAN.md
│   │   ├── VOICE_MEMORY_IMPLEMENTATION_COMPLETE.md
│   │   ├── VOICE_MEMORY_INTEGRATION_SUMMARY.md
│   │   └── VOICE_TOOLS_INTEGRATION_PLAN.md
│   │
│   └── DIRECTORY_STRUCTURE.md     # This file
│
├── data/                          # Application data
│   ├── memory/                    # Memory system data
│   │   ├── audio_metadata.json
│   │   ├── conversations.json
│   │   ├── transcription_quality.json
│   │   ├── user_preferences.json
│   │   ├── visual_context.json
│   │   ├── voice_command_patterns.json
│   │   ├── voice_sessions.json
│   │   └── wake_word_stats.json
│   └── voice_conversations.db     # Main conversation database
│
├── tools/                         # Application modules
│   ├── __init__.py
│   ├── analysis/                  # Analysis tools
│   ├── config/                    # Configuration tools
│   ├── memory/                    # Memory system
│   │   ├── README.md
│   │   ├── VOICE_MEMORY_MANAGER.md
│   │   ├── __init__.py
│   │   ├── voice_context_memory.py
│   │   ├── voice_conversation_memory.py
│   │   ├── voice_memory_integration_example.py
│   │   ├── voice_memory_manager.py
│   │   ├── voice_memory_manager_example.py
│   │   └── voice_memory_setup.py
│   └── vision/                    # Vision integration
│
├── tests/                         # All test files
│   ├── debug/                     # Debug and test scripts
│   │   ├── ai_provider_usage_example.py
│   │   ├── debug_memory.py
│   │   ├── example_voice_conversations.db
│   │   ├── test_ai_integration.py
│   │   ├── test_ai_providers.py
│   │   ├── test_ai_providers_simple.py
│   │   ├── test_integration.py
│   │   ├── test_memory_dialog.py
│   │   ├── test_memory_dialog_fix.py
│   │   ├── test_openai_quick.py
│   │   ├── test_openai_status.py
│   │   └── test_voice_conversations.db
│   │
│   ├── integration/               # Integration tests
│   └── tools/                     # Tool-specific tests
│       └── memory/
│           └── test_voice_conversation_memory.py
│
├── icons/                         # Application icons
│   ├── create_icon.py
│   ├── whisper-icon.png
│   └── whisper-icon.svg
│
└── venv/                          # Python virtual environment
```

## Key Changes Made

1. **Documentation Organized**: All markdown files moved to `docs/` with subdirectories for planning and implementation documents
2. **Tests Consolidated**: All test files moved to `tests/debug/` directory
3. **Root Directory Cleaned**: Only essential files remain in root (main scripts, launchers, README, LICENSE)
4. **Clear Hierarchy**: Logical grouping makes navigation easier

## Benefits

- **Cleaner Root**: Essential files are immediately visible
- **Better Organization**: Related files are grouped together
- **Easier Navigation**: Clear structure helps developers find files quickly
- **Maintainability**: Organized structure makes the project easier to maintain