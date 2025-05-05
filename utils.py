#utils.py
import os
import fitz
import glob
from typing import List

def load_data(data_dir: str="data") -> List[str]:
    """ Load texts from documents """
    documents = []
    for filename in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(filename, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    
    for filename in glob.glob(os.path.join(data_dir, "*.pdf")):
        try:
            with fitz.open(filename) as doc:
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                if text.strip():
                    documents.append(text)
                else:
                    print(f"Could not extract text from {filename}")
        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")
    return documents

def get_source_names(data_dir: str="data") -> List[str]:
    """ Return source filenames """
    source_name = []
    for filename in glob.glob(os.path.join(data_dir, "*.txt")):
        source_name.append(os.path.basename(filename))
    
    for filename in glob.glob(os.path.join(data_dir, "*.pdf")):
        source_name.append(os.path.basename(filename))
    
    return source_name

def preprocess_text(text: str) -> str:
    """Text preprocessing"""
    text = text.lower().strip()
    return ' '.join(text.split())

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 100) -> List[str]:
    """Simple Chunking"""
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks
