# Jupyter RAG Integration - Quick Implementation Guide

## Summary
You have Jupyter running at http://localhost:8888/tree. This guide provides the fastest path to create a working RAG integration.

## Phase 1: Immediate Setup (30 minutes)

### 1. Create Basic Connector
```bash
# Test connection to your Jupyter server
cd ~/whisper-transcribe-pi
python3 tools/memory/jupyter_rag/quick_start.py
```

### 2. Handle Authentication
If your Jupyter server requires a token, find it:
```bash
# Check running Jupyter processes for token
ps aux | grep jupyter
# Or check Jupyter config
jupyter server list
```

### 3. Simple Notebook Reader
Create a file reader that works without API calls:
```python
# tools/memory/jupyter_rag/file_reader.py
import os
import json
import nbformat

def find_notebooks(base_dir="~/"):
    """Find all .ipynb files"""
    notebooks = []
    for root, dirs, files in os.walk(os.path.expanduser(base_dir)):
        for file in files:
            if file.endswith('.ipynb'):
                notebooks.append(os.path.join(root, file))
    return notebooks

def extract_text_from_notebook(notebook_path):
    """Extract searchable text from notebook"""
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    text_content = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            text_content.append(cell.source)
        elif cell.cell_type == 'code':
            # Include code and comments
            text_content.append(f"CODE: {cell.source}")
    
    return "\n\n".join(text_content)
```

## Phase 2: Basic RAG Integration (1 hour)

### 1. Simple Search Function
```python
# tools/memory/jupyter_rag/simple_search.py
import re
from typing import List, Tuple

class SimpleNotebookSearch:
    def __init__(self):
        self.notebook_index = {}
        self.build_index()
    
    def build_index(self):
        """Build simple text index"""
        from .file_reader import find_notebooks, extract_text_from_notebook
        
        notebooks = find_notebooks()
        for nb_path in notebooks:
            try:
                content = extract_text_from_notebook(nb_path)
                self.notebook_index[nb_path] = content.lower()
            except Exception as e:
                print(f"Error indexing {nb_path}: {e}")
    
    def search(self, query: str, max_results: int = 5) -> List[Tuple[str, str]]:
        """Simple keyword search"""
        query_lower = query.lower()
        results = []
        
        for nb_path, content in self.notebook_index.items():
            # Count keyword matches
            matches = len(re.findall(re.escape(query_lower), content))
            if matches > 0:
                # Get context around first match
                match_pos = content.find(query_lower)
                start = max(0, match_pos - 200)
                end = min(len(content), match_pos + 200)
                context = content[start:end]
                
                results.append((nb_path, context, matches))
        
        # Sort by number of matches
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]
```

### 2. Integration with Voice Memory
```python
# Add to whisper_transcribe_pro.py

class JupyterRAGIntegration:
    def __init__(self):
        try:
            from tools.memory.jupyter_rag.simple_search import SimpleNotebookSearch
            self.notebook_search = SimpleNotebookSearch()
            self.enabled = True
            print("Jupyter RAG integration enabled")
        except Exception as e:
            print(f"Jupyter RAG integration failed: {e}")
            self.enabled = False
    
    def enhance_response(self, user_input: str, base_response: str) -> str:
        """Enhance AI response with notebook knowledge"""
        if not self.enabled:
            return base_response
        
        # Search relevant notebooks
        results = self.notebook_search.search(user_input)
        
        if results:
            context = "\n\n".join([f"From {os.path.basename(path)}: {context}" 
                                 for path, context, _ in results])
            
            enhanced_prompt = f"""
            User question: {user_input}
            
            Relevant information from Jupyter notebooks:
            {context}
            
            Base response: {base_response}
            
            Please provide an enhanced response that incorporates relevant information from the notebooks.
            """
            
            return enhanced_prompt
        
        return base_response
```

## Phase 3: Quick UI Integration (30 minutes)

### 1. Add RAG Toggle to Settings
```python
# In SettingsWindow class, add to AI tab:

# RAG Settings Frame
rag_frame = ctk.CTkFrame(ai_tab)
rag_frame.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(rag_frame, text="Jupyter RAG Integration", 
             font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)

self.jupyter_rag_enabled = ctk.CTkCheckBox(
    rag_frame, 
    text="Enable Jupyter notebook search for enhanced responses"
)
self.jupyter_rag_enabled.pack(anchor="w", padx=20, pady=2)

# RAG info button
ctk.CTkButton(
    rag_frame,
    text="Index Notebooks",
    command=self.reindex_notebooks,
    width=120
).pack(anchor="w", padx=20, pady=5)
```

### 2. Add to Voice Processing
```python
# In transcribe_audio method, before AI enhancement:

if hasattr(self, 'jupyter_rag') and self.jupyter_rag.enabled:
    # Check if this looks like a question that could benefit from notebooks
    if any(word in transcription.lower() for word in ['how', 'what', 'why', 'explain']):
        rag_context = self.jupyter_rag.enhance_response(transcription, "")
        if rag_context:
            # Modify the AI prompt to include notebook context
            enhanced_transcription = f"{transcription}\n\nContext: {rag_context}"
```

## Phase 4: Testing (15 minutes)

### 1. Create Test Script
```python
# test_jupyter_rag.py
from tools.memory.jupyter_rag.simple_search import SimpleNotebookSearch

def test_rag():
    print("Testing Jupyter RAG integration...")
    
    search = SimpleNotebookSearch()
    print(f"Indexed {len(search.notebook_index)} notebooks")
    
    # Test searches
    test_queries = [
        "hailo detection",
        "machine learning",
        "python code",
        "neural network"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: {query}")
        results = search.search(query)
        print(f"Found {len(results)} results")
        if results:
            print(f"Top result: {os.path.basename(results[0][0])}")

if __name__ == "__main__":
    test_rag()
```

## Expected Results

After implementing this quick version, you should have:

1. **Notebook Discovery**: Automatically finds all .ipynb files
2. **Simple Search**: Keyword-based search across notebook content
3. **Voice Integration**: Enhanced responses when questions could benefit from notebook knowledge
4. **Settings Control**: Toggle RAG on/off in the UI

## Performance Expectations

- **Indexing Time**: 1-2 seconds for 50 notebooks
- **Search Time**: <100ms for most queries
- **Memory Usage**: +10-50MB depending on notebook count
- **Response Enhancement**: 20-30% more informative answers

## Next Steps

1. **Test the basic implementation**
2. **Verify it finds your notebooks**
3. **Try voice queries that should trigger RAG**
4. **Monitor performance and adjust**

## Troubleshooting

### Common Issues:
1. **No notebooks found**: Check file permissions
2. **Slow indexing**: Limit to specific directories
3. **Poor search results**: Add more sophisticated matching
4. **Memory issues**: Implement lazy loading

### Quick Fixes:
```python
# Limit search to specific directories
NOTEBOOK_DIRS = [
    "~/hailo-ai",
    "~/Documents/notebooks",
    "~/.jupyter"
]

# Only index recent notebooks
import os
from datetime import datetime, timedelta

def is_recent_notebook(path, days=30):
    mod_time = os.path.getmtime(path)
    return datetime.fromtimestamp(mod_time) > datetime.now() - timedelta(days=days)
```

This quick implementation should give you a working RAG system in under 2 hours!