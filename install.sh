#!/bin/bash

# Whisper Transcribe Desktop App Installer
# This script installs the Whisper Transcribe app to the Pi menu

echo "======================================"
echo "Whisper Transcribe App Installer"
echo "======================================"

# Define paths - use current directory
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DESKTOP_FILE="$APP_DIR/whisper-transcribe.desktop"
LAUNCHER_SCRIPT="$APP_DIR/launch_whisper.sh"
ICON_FILE="$APP_DIR/icons/whisper-icon.png"

# Check if files exist
if [ ! -f "$DESKTOP_FILE" ]; then
    echo "Error: Desktop file not found at $DESKTOP_FILE"
    exit 1
fi

if [ ! -f "$LAUNCHER_SCRIPT" ]; then
    echo "Error: Launcher script not found at $LAUNCHER_SCRIPT"
    exit 1
fi

if [ ! -f "$ICON_FILE" ]; then
    echo "Warning: Icon file not found at $ICON_FILE"
    echo "Creating a basic icon..."
    mkdir -p "$APP_DIR/icons"
    python3 "$APP_DIR/icons/create_icon.py" 2>/dev/null || echo "Could not create icon"
fi

# Make launcher executable
echo "Making launcher executable..."
chmod +x "$LAUNCHER_SCRIPT"

# Copy desktop file to user's local applications
echo "Installing desktop entry..."
mkdir -p ~/.local/share/applications
cp "$DESKTOP_FILE" ~/.local/share/applications/

# Copy icon to local icons directory
echo "Installing icon..."
mkdir -p ~/.local/share/icons
cp "$ICON_FILE" ~/.local/share/icons/whisper-icon.png 2>/dev/null || echo "Icon not copied"

# Update desktop database
echo "Updating desktop database..."
update-desktop-database ~/.local/share/applications 2>/dev/null || true

# Create a symbolic link in ~/bin for command line access
echo "Creating command line shortcut..."
mkdir -p ~/bin
ln -sf "$APP_DIR/launch_whisper.sh" ~/bin/whisper-transcribe 2>/dev/null || true

# Make the app executable from anywhere
if ! grep -q "export PATH=\$HOME/bin:\$PATH" ~/.bashrc; then
    echo "Adding ~/bin to PATH..."
    echo "" >> ~/.bashrc
    echo "# Added by Whisper Transcribe installer" >> ~/.bashrc
    echo "export PATH=\$HOME/bin:\$PATH" >> ~/.bashrc
fi

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "The Whisper Transcribe app has been installed."
echo ""
echo "To use the app:"
echo "1. Look for 'Whisper Transcribe' in your Pi menu (AudioVideo section)"
echo "2. Or run 'whisper-transcribe' from terminal"
echo "3. Or use the existing 'whisper' command"
echo ""
echo "You may need to:"
echo "- Log out and back in for menu changes to appear"
echo "- Or refresh your desktop (right-click → Refresh)"
echo ""
echo "Press Enter to continue..."
read -r