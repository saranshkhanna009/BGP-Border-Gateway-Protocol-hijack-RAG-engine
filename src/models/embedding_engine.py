

from typing import List
import numpy as np

class BGPPlaybookEmbedder:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            # Loads the pre-trained transformer model (automatically downloads on first execution)
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' package is missing.\n"
                "Please run: pip install sentence-transformers"
            )

    def embed_query(self, text: str) -> List[float]:
        
        if not text.strip():
            raise ValueError("Query text cannot be empty.")
            
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        
        if not texts:
            return []
            
        # Strip and validate input list
        cleaned_texts = [t.strip() for t in texts if t.strip()]
        if not cleaned_texts:
            raise ValueError("All provided document chunks were empty.")
            
        embeddings = self.model.encode(cleaned_texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def embedding_dimension(self) -> int:
        
        # all-MiniLM-L6-v2 outputs 384 dimensions; BGE-large outputs 1024 dimensions
        return self.model.get_sentence_embedding_dimension()