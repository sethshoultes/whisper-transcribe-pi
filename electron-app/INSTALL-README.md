# Whisper Transcribe Electron App - Installation Guide

This directory contains the Electron app version of Whisper Transcribe with **automatic dependency management** and **one-click installation**.

## 🚀 Quick Start (Recommended)

### Option 1: Automatic Installer (Easiest)
```bash
# Navigate to the electron-app directory
cd electron-app

# Run the automatic installer
./install.sh
```

This will:
- ✅ Check system requirements
- ✅ Verify Python installation  
- ✅ Create bundled Python environment
- ✅ Install all dependencies automatically
- ✅ Download AI models
- ✅ Set up the Electron app

### Option 2: Manual Step-by-Step
```bash
# 1. Install Node.js dependencies
npm install

# 2. Create Python bundle with dependencies
npm run setup-python

# 3. Start the app
npm start
```

## 📋 Requirements

- **Node.js 16+** (Download from [nodejs.org](https://nodejs.org/))
- **Python 3.8+** (Download from [python.org](https://python.org/downloads/))
- **macOS 10.14+** / **Windows 10+** / **Linux (Ubuntu 18.04+)**
- **~2GB free disk space** (for AI models and dependencies)

## 🛠 Installation Options

### For macOS
```bash
# Install Python via Homebrew (recommended)
brew install python@3.11

# Or download from python.org
open https://www.python.org/downloads/
```

### For Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. ✅ Check "Add Python to PATH" during installation
3. Open Command Prompt and run the installer

### For Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv nodejs npm
```

## 📁 Project Structure

```
electron-app/
├── install.sh              # 🚀 One-click installer script
├── installer.js            # 📦 Main installer logic
├── main.js                 # 🖥️ Main Electron process
├── dependency-manager.js   # 🔧 Automatic dependency handling
├── error-handler.js        # ❌ User-friendly error recovery
├── create-bundle-env.js    # 🐍 Python environment bundler
├── backend/                # 🐍 Python Flask server
│   ├── server.py
│   └── requirements.txt
└── python-bundle/          # 📦 Self-contained Python (auto-created)
    └── venv/
```

## 🎯 How It Works

1. **Dependency Detection**: Automatically finds your Python installation
2. **Bundle Creation**: Creates a self-contained Python environment
3. **Package Installation**: Installs all required packages (Flask, Whisper, etc.)
4. **Model Download**: Pre-downloads AI models for offline use
5. **Error Recovery**: Provides helpful solutions when things go wrong

## 🔧 Troubleshooting

### "Python not found"
```bash
# Install Python
brew install python@3.11  # macOS
# or visit python.org

# Verify installation
python3 --version
```

### "Permission denied"
```bash
# Fix permissions
chmod +x install.sh
```

### "Module not found" 
The app will automatically detect and install missing packages. If this fails:
```bash
# Manual installation
pip3 install --user flask flask-cors openai-whisper scipy numpy
```

### "Disk space" error
Free up at least 2GB of space:
- Empty Downloads folder
- Empty Trash
- Remove unused applications

## 🚀 Running the App

### Development Mode
```bash
npm start
```

### Build Distributable App
```bash
npm run dist
```

The built app will be in the `dist/` folder.

## 📊 Features

- ✅ **Zero Manual Setup**: Installs everything automatically
- ✅ **Self-Contained**: Bundles Python environment
- ✅ **Error Recovery**: Helpful error messages with solutions
- ✅ **Cross-Platform**: Works on macOS, Windows, Linux
- ✅ **Offline Ready**: Downloads models during installation
- ✅ **User Permissions**: Asks before installing anything

## 🔍 Advanced Usage

### Clean Installation
```bash
npm run clean      # Remove all generated files
./install.sh       # Reinstall everything
```

### Python Bundle Only
```bash
npm run setup-python
```

### Custom Python Path
The dependency manager automatically finds the best Python installation, checking:
1. Bundled Python (if exists)
2. System Python 3.11, 3.12, 3.10
3. Generic python3/python commands

## 📝 Logs

Installation and runtime logs are saved to:
- **Installation**: `electron-app/install.log`
- **Runtime**: `~/Library/Application Support/whisper-transcribe/whisper-app.log` (macOS)

## 🆘 Support

If you encounter issues:

1. **Check the logs** (locations above)
2. **Run the installer again**: `./install.sh`
3. **Manual reset**: `npm run clean && ./install.sh`
4. **Report issues**: [GitHub Issues](https://github.com/sethshoultes/whisper-transcribe-pi/issues)

## 🎉 Success!

After installation, you should see:
```
🎉 Installation Complete!
========================

You can now run Whisper Transcribe with:
  npm start

Or build the distributable app with:
  npm run dist
```

The app will start with a floating window for speech-to-text transcription, with automatic clipboard integration and error recovery built-in.