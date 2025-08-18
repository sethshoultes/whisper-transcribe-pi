#!/usr/bin/env python3
"""
Quick start script to connect to existing Jupyter server and test RAG functionality
"""

import requests
import json
from typing import List, Dict, Any

class JupyterServerConnector:
    """Connect to running Jupyter server"""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
    def test_connection(self) -> bool:
        """Test if Jupyter server is accessible"""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_notebooks(self, path: str = "") -> List[Dict[str, Any]]:
        """List available notebooks"""
        try:
            response = requests.get(f"{self.api_url}/contents/{path}")
            if response.status_code == 200:
                contents = response.json()
                notebooks = [
                    item for item in contents.get('content', [])
                    if item['type'] == 'notebook'
                ]
                return notebooks
            return []
        except Exception as e:
            print(f"Error listing notebooks: {e}")
            return []
    
    def get_notebook_content(self, path: str) -> Dict[str, Any]:
        """Get notebook content"""
        try:
            response = requests.get(f"{self.api_url}/contents/{path}")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"Error getting notebook: {e}")
            return {}

def main():
    """Test Jupyter connection and list notebooks"""
    
    print("Connecting to Jupyter server at http://localhost:8888...")
    connector = JupyterServerConnector()
    
    if connector.test_connection():
        print("✓ Successfully connected to Jupyter server!")
        
        print("\nListing available notebooks...")
        notebooks = connector.list_notebooks()
        
        if notebooks:
            print(f"Found {len(notebooks)} notebooks:")
            for nb in notebooks:
                print(f"  - {nb['name']} ({nb['path']})")
        else:
            print("No notebooks found or unable to list notebooks.")
            print("Note: You may need to configure authentication if the server requires a token.")
    else:
        print("✗ Could not connect to Jupyter server.")
        print("Please ensure Jupyter is running at http://localhost:8888")
        print("You can start it with: hailo jupyter")

if __name__ == "__main__":
    main()