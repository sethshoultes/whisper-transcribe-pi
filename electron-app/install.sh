#!/bin/bash

# Whisper Transcribe Standalone Installer
echo "🚀 Whisper Transcribe Standalone Installer"
echo "==========================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed!"
    echo "Please install Node.js from https://nodejs.org/"
    echo ""
    echo "For macOS with Homebrew:"
    echo "  brew install node"
    echo ""
    echo "Then run this installer again."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this installer from the electron-app directory"
    exit 1
fi

# Run the Node.js installer
echo "Starting Node.js installer..."
node installer.js

# Check if installation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "To start Whisper Transcribe:"
    echo "  npm start"
    echo ""
    echo "To build a distributable app:"
    echo "  npm run dist"
else
    echo ""
    echo "💥 Installation failed!"
    echo "Check the install.log file for details."
    exit 1
fi