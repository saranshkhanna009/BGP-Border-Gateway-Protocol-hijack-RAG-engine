# src/data/document_parser.py

import os
import re
from typing import List, Dict

class OperationalDocumentParser:
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def normalize_text(self, text: str) -> str:
        
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def generate_chunks(self, document_id: str, raw_content: str) -> List[Dict[str, any]]:
        
        cleaned_text = self.normalize_text(raw_content)
        words = cleaned_text.split(' ')
        
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        # Guard against zero/negative steps to prevent infinite loop execution loops
        if step <= 0:
            step = self.chunk_size // 2

        for i in range(0, len(words), step):
            window_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(window_words)
            
            # Record structural location metadata alongside actual extracted strings
            chunks.append({
                "document_id": document_id,
                "chunk_index": len(chunks),
                "text_content": chunk_text,
                "word_count": len(window_words)
            })
            
            if i + self.chunk_size >= len(words):
                break
                
        return chunks