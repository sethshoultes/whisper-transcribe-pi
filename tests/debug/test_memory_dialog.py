#!/usr/bin/env python3
"""Test script to verify memory dialog functionality"""

import tkinter as tk
import customtkinter as ctk
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_transcribe_pro import WhisperTranscribePro

def test_memory_dialog():
    """Test opening the memory dialog"""
    print("Starting test application...")
    
    # Create test app
    app = WhisperTranscribePro()
    
    # Schedule memory dialog to open after 1 second
    def open_memory():
        print("Opening memory dialog...")
        app.open_memory_menu()
        print("Memory dialog opened successfully!")
    
    app.after(1000, open_memory)
    
    # Add a button to manually open memory dialog
    test_btn = ctk.CTkButton(
        app,
        text="Test Memory Dialog",
        command=app.open_memory_menu,
        width=150,
        height=40
    )
    test_btn.pack(pady=20)
    
    print("Test application ready. Memory dialog will open in 1 second...")
    print("You can also click 'Test Memory Dialog' button to open it manually.")
    
    # Run the app
    app.mainloop()

if __name__ == "__main__":
    test_memory_dialog()