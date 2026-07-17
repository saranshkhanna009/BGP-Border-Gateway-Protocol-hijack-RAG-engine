

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class BGPVectorStore:
    """
    Manages semantic vector store operations (indexing, metadata schema definition, and search)
    using local-mode Qdrant Client.
    """
    def __init__(self, collection_name: str = "bgp_playbooks", vector_size: int = 384):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize fully in-memory, perfect for standalone runs/CI tests
        self.client = QdrantClient(":memory:")
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Guarantees the target collection is set up and configured correctly."""
        # Using check to prevent raising errors if already created
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    def upsert_playbook_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Saves parsed document chunk content, matching high-dimensional vector positions,
        and operational metadata into Qdrant.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Mismatched length: Chunks and vector embeddings must align.")

        points = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=index,
                    vector=embedding,
                    payload={
                        "document_id": chunk.get("document_id", "unknown"),
                        "chunk_index": chunk.get("chunk_index", index),
                        "text_content": chunk.get("text_content", ""),
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

    def search_similar_playbooks(self, query_vector: List[float], limit: int = 2) -> List[Dict[str, Any]]:
        """
        Queries the vector index using cosine similarity to pull operational context matching
        a target query vector.
        """
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points

        output = []
        for hit in search_results:
            output.append({
                "score": hit.score,
                "document_id": hit.payload.get("document_id"),
                "text_content": hit.payload.get("text_content")
            })
        return output