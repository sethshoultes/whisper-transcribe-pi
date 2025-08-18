#!/bin/bash

echo "Building Whisper Transcribe Mac App..."

# Clean previous builds
rm -rf electron-app/dist

# Ensure we have ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
fi

# Build the Electron app
cd electron-app
npm run dist

echo "Build complete! The app is in electron-app/dist/"
echo "You can find Whisper Transcribe.app in that folder."