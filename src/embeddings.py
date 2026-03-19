"""
MediRAG - Embedding Generation Module
Generates vector embeddings for disease documents using sentence-transformers.
Uses the all-MiniLM-L6-v2 model for fast, high-quality embeddings.
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


# Default embedding model
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Cache directory for embeddings
CACHE_DIR = "embeddings_cache"


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating and caching embeddings.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier for sentence-transformers
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  ✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], normalize: bool = True, 
               batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            normalize: Whether to L2-normalize embeddings (required for cosine similarity)
            batch_size: Batch size for encoding
            show_progress: Whether to show a progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return np.array(embeddings, dtype=np.float32)
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query text to embed
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            numpy array of shape (1, embedding_dim)
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return np.array(embedding, dtype=np.float32)


def generate_document_embeddings(
    documents: List[Dict],
    model: Optional[EmbeddingModel] = None,
    cache_path: Optional[str] = None
) -> np.ndarray:
    """
    Generate embeddings for disease documents, with optional caching.
    
    Args:
        documents: List of document dicts (must have 'text' key)
        model: EmbeddingModel instance (created if not provided)
        cache_path: Path to cache embeddings (skips if None)
        
    Returns:
        numpy array of embeddings, shape (num_docs, embedding_dim)
    """
    # Check cache first
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        print(f"  ✓ Loaded {cached.shape[0]} cached embeddings")
        return cached
    
    # Create model if not provided
    if model is None:
        model = EmbeddingModel()
    
    # Extract text from documents
    texts = [doc['text'] for doc in documents]
    
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts)
    print(f"  ✓ Generated embeddings: shape {embeddings.shape}")
    
    # Cache embeddings if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"  ✓ Cached embeddings to {cache_path}")
    
    return embeddings


if __name__ == "__main__":
    from src.preprocessing import create_disease_documents
    
    # Create disease documents
    documents = create_disease_documents()
    
    # Generate embeddings
    cache_file = os.path.join(CACHE_DIR, "disease_embeddings.pkl")
    embeddings = generate_document_embeddings(documents, cache_path=cache_file)
    
    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Test query encoding
    model = EmbeddingModel()
    query_emb = model.encode_query("fever headache joint pain")
    print(f"\nQuery embedding shape: {query_emb.shape}")
    print(f"Query embedding (first 10 dims): {query_emb[0][:10]}")
