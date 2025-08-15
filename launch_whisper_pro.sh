#!/bin/bash

# Whisper Transcribe Pro Launcher Script
# Launches the enhanced Pro version with modern UI

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    # Create notification that setup is needed
    zenity --error --text="First time setup required. Please run:\n\ncd $SCRIPT_DIR\n./setup_pro.sh" 2>/dev/null || \
    notify-send "Setup Required" "Please run setup_pro.sh first" || \
    xmessage "First time setup required. Please run setup_pro.sh"
    exit 1
fi

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Check if dependencies are installed
if ! python -c "import customtkinter" 2>/dev/null; then
    zenity --info --text="Installing dependencies, please wait..." 2>/dev/null &
    pip install -r requirements.txt > /tmp/whisper_install.log 2>&1
fi

# Kill any existing instances
pkill -f whisper_transcribe_pro.py 2>/dev/null

# Launch the Pro version without terminal window
nohup python3 "$SCRIPT_DIR/whisper_transcribe_pro.py" > /tmp/whisper_pro.log 2>&1 &

# Give it a moment to start
sleep 1

# Check if it started successfully
if pgrep -f whisper_transcribe_pro.py > /dev/null; then
    exit 0
else
    # Show error if it failed to start
    zenity --error --text="Failed to start Whisper Transcribe Pro.\nCheck /tmp/whisper_pro.log for details." 2>/dev/null || \
    notify-send "Error" "Failed to start Whisper Transcribe Pro" || \
    xmessage "Failed to start Whisper Transcribe Pro"
    exit 1
fi