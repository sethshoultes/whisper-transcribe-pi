#!/usr/bin/env python3
"""
Create a simple microphone icon for Whisper Transcribe
"""

svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <!-- Background circle -->
  <circle cx="128" cy="128" r="120" fill="#4CAF50" stroke="#388E3C" stroke-width="4"/>
  
  <!-- Microphone body -->
  <rect x="108" y="60" width="40" height="80" rx="20" fill="white"/>
  
  <!-- Microphone stand -->
  <path d="M 128 140 L 128 180" stroke="white" stroke-width="8" stroke-linecap="round"/>
  
  <!-- Microphone base -->
  <path d="M 100 180 L 156 180" stroke="white" stroke-width="8" stroke-linecap="round"/>
  
  <!-- Recording arc -->
  <path d="M 88 100 Q 88 140 108 150" 
        stroke="white" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M 168 100 Q 168 140 148 150" 
        stroke="white" stroke-width="4" fill="none" stroke-linecap="round"/>
  
  <!-- Recording dot indicator -->
  <circle cx="128" cy="90" r="8" fill="#FF5252"/>
</svg>'''

# Save as SVG
with open('/home/sethshoultes/ai-assistant-project/icons/whisper-icon.svg', 'w') as f:
    f.write(svg_content)

print("SVG icon created")

# Convert to PNG using rsvg-convert if available
import subprocess
import os

try:
    # Check if rsvg-convert is available
    result = subprocess.run(['which', 'rsvg-convert'], capture_output=True)
    if result.returncode == 0:
        # Convert SVG to PNG
        subprocess.run([
            'rsvg-convert',
            '-w', '256',
            '-h', '256',
            '/home/sethshoultes/ai-assistant-project/icons/whisper-icon.svg',
            '-o', '/home/sethshoultes/ai-assistant-project/icons/whisper-icon.png'
        ])
        print("PNG icon created")
    else:
        print("rsvg-convert not found, installing...")
        subprocess.run(['sudo', 'apt', 'install', '-y', 'librsvg2-bin'], check=False)
        # Try again after install
        subprocess.run([
            'rsvg-convert',
            '-w', '256',
            '-h', '256',
            '/home/sethshoultes/ai-assistant-project/icons/whisper-icon.svg',
            '-o', '/home/sethshoultes/ai-assistant-project/icons/whisper-icon.png'
        ])
        print("PNG icon created")
except Exception as e:
    print(f"Could not create PNG: {e}")
    print("SVG icon is available")