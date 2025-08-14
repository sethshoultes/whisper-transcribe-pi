#!/bin/bash

# Whisper Transcribe Launcher Script
# This script launches the Whisper Transcribe application

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists, if not create it
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
    
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install openai-whisper sounddevice numpy scipy tkinter 2>/dev/null || true
else
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Check if another instance is running
if pgrep -f "whisper_transcribe.py" > /dev/null; then
    echo "Whisper is already running. Stopping existing instance..."
    pkill -f "whisper_transcribe.py"
    sleep 1
fi

# Launch the Whisper transcription app
echo "Starting Whisper Transcribe..."
python3 "$SCRIPT_DIR/whisper_transcribe.py"

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo "Error launching Whisper Transcribe"
    read -p "Press Enter to close..."
fi