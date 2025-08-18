#!/bin/bash

echo "Setting up Whisper Transcribe dependencies..."

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 is not installed. Please install it first."
    echo "You can install it using Homebrew: brew install python@3.11"
    exit 1
fi

echo "Installing required Python packages..."

# Install packages globally for Python 3.11
/usr/local/bin/python3.11 -m pip install --upgrade pip
/usr/local/bin/python3.11 -m pip install flask flask-cors openai-whisper scipy numpy<2

echo "Dependencies installed successfully!"
echo "You can now run Whisper Transcribe."