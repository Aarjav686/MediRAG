"""
MediRAG - FAISS Vector Store Module
Manages the FAISS index for efficient similarity search over disease embeddings.
Uses IndexFlatIP (inner product) on L2-normalized vectors for cosine similarity.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple


# Default paths for persisting the index
DEFAULT_INDEX_DIR = "faiss_index"
INDEX_FILENAME = "disease_index.faiss"
METADATA_FILENAME = "index_metadata.json"


class VectorStore:
    """
    FAISS-based vector store for disease document retrieval.
    Stores embeddings and associated metadata for similarity search.
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        self.embedding_dim: int = 0
    
    def build_index(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        """
        Build a FAISS index from embeddings and store associated documents.
        Uses IndexFlatIP (inner product) — when used with L2-normalized vectors,
        this computes cosine similarity.
        
        Args:
            embeddings: numpy array of shape (num_docs, embedding_dim), must be L2-normalized
            documents: List of document dicts corresponding to each embedding
        """
        assert len(embeddings) == len(documents), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents"
        
        self.embedding_dim = embeddings.shape[1]
        self.documents = documents
        
        # Create FAISS index using inner product (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to the index
        self.index.add(embeddings)
        
        print(f"  ✓ Built FAISS index with {self.index.ntotal} vectors, dim={self.embedding_dim}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search the index for the most similar documents to a query embedding.
        
        Args:
            query_embedding: numpy array of shape (1, embedding_dim), must be L2-normalized
            top_k: Number of top results to return
            
        Returns:
            List of (document_dict, similarity_score) tuples, sorted by score descending
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first or load an existing index.")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Build results list
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_dir: Directory to save the index files
        """
        if self.index is None:
            raise ValueError("No index to save. Build an index first.")
        
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(index_dir, INDEX_FILENAME)
        faiss.write_index(self.index, index_path)
        
        # Save document metadata as JSON
        metadata_path = os.path.join(index_dir, METADATA_FILENAME)
        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_documents': len(self.documents),
            'documents': self.documents,
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved FAISS index to {index_path}")
        print(f"  ✓ Saved metadata to {metadata_path}")
    
    def load(self, index_dir: str = DEFAULT_INDEX_DIR) -> bool:
        """
        Load a FAISS index and metadata from disk.
        
        Args:
            index_dir: Directory containing the index files
            
        Returns:
            True if loaded successfully, False if files don't exist
        """
        index_path = os.path.join(index_dir, INDEX_FILENAME)
        metadata_path = os.path.join(index_dir, METADATA_FILENAME)
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"  ✗ Index files not found in {index_dir}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load document metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        self.documents = metadata['documents']
        
        print(f"  ✓ Loaded FAISS index: {self.index.ntotal} vectors, dim={self.embedding_dim}")
        return True


def build_and_save_vector_store(
    documents: List[Dict],
    embeddings: np.ndarray,
    index_dir: str = DEFAULT_INDEX_DIR
) -> VectorStore:
    """
    Convenience function to build a vector store and save it to disk.
    
    Args:
        documents: List of disease document dicts
        embeddings: Corresponding embeddings array
        index_dir: Directory to save the index
        
    Returns:
        Built VectorStore instance
    """
    store = VectorStore()
    store.build_index(embeddings, documents)
    store.save(index_dir)
    return store


if __name__ == "__main__":
    from src.preprocessing import create_disease_documents
    from src.embeddings import EmbeddingModel, generate_document_embeddings
    
    # Create documents and embeddings
    documents = create_disease_documents()
    embeddings = generate_document_embeddings(documents)
    
    # Build and save vector store
    store = build_and_save_vector_store(documents, embeddings)
    
    # Test search
    model = EmbeddingModel()
    query_emb = model.encode_query("fever headache joint pain")
    
    print("\n--- Search Results for 'fever headache joint pain' ---")
    results = store.search(query_emb, top_k=5)
    for doc, score in results:
        print(f"  {doc['disease']}: {score:.4f}")
    
    # Test load
    print("\n--- Testing load from disk ---")
    store2 = VectorStore()
    store2.load()
    results2 = store2.search(query_emb, top_k=3)
    for doc, score in results2:
        print(f"  {doc['disease']}: {score:.4f}")
