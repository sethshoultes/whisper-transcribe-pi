#!/bin/bash

# Whisper Transcribe Pro - One-Click Setup Script
# Sets up everything needed for the Pro version

echo "======================================"
echo "Whisper Transcribe Pro Setup"
echo "======================================"
echo ""
echo "Setting up the enhanced version with:"
echo "• Modern UI with themes"
echo "• Hailo AI integration"
echo "• Advanced settings"
echo "• Professional features"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION found"

# Install system dependencies
echo ""
echo "Installing system dependencies..."
if ! dpkg -l | grep -q portaudio19-dev; then
    sudo apt update
    sudo apt install -y portaudio19-dev
    echo "✓ Audio dependencies installed"
else
    echo "✓ Audio dependencies already installed"
fi

# Create virtual environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate and install Python packages
source venv/bin/activate

echo ""
echo "Installing Python packages..."
echo "This may take a few minutes on first install..."

# Install packages with progress
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"Collecting"* ]] || [[ "$line" == *"Installing"* ]]; then
        echo "  $line"
    fi
done

echo "✓ Python packages installed"

# Check for Hailo
echo ""
echo "Checking for Hailo AI hardware..."
if which hailortcli > /dev/null 2>&1; then
    echo "✓ Hailo AI detected - enhanced features will be available!"
else
    echo "ℹ Hailo not detected - app will work without AI enhancements"
fi

# Make scripts executable
chmod +x launch_whisper_pro.sh
chmod +x whisper_transcribe_pro.py

# Install desktop entry
echo ""
echo "Installing desktop entry..."
mkdir -p ~/.local/share/applications
cp whisper-transcribe-pro.desktop ~/.local/share/applications/
update-desktop-database ~/.local/share/applications 2>/dev/null || true
echo "✓ Desktop entry installed"

# Create desktop shortcut (optional)
if [ -d "$HOME/Desktop" ]; then
    cp whisper-transcribe-pro.desktop "$HOME/Desktop/"
    chmod +x "$HOME/Desktop/whisper-transcribe-pro.desktop"
    gio set "$HOME/Desktop/whisper-transcribe-pro.desktop" metadata::trusted true 2>/dev/null || true
    echo "✓ Desktop shortcut created"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Whisper Transcribe Pro is ready to use!"
echo ""
echo "Launch options:"
echo "1. Click 'Whisper Transcribe Pro' in your applications menu"
echo "2. Double-click the desktop shortcut"
echo "3. Run: ./launch_whisper_pro.sh"
echo ""
echo "The app will start without a terminal window."
echo ""
read -p "Press Enter to launch the app now, or Ctrl+C to exit..."

# Launch the app
./launch_whisper_pro.sh