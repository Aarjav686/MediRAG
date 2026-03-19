"""
MediRAG - Retriever Module
Handles query preprocessing, FAISS similarity search, and result re-ranking.
Combines vector similarity with symptom overlap and severity scoring for
more accurate disease prediction.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


# Common symptom name normalizations (user-friendly -> dataset format)
SYMPTOM_ALIASES = {
    "stomach ache": "stomach_pain",
    "belly ache": "belly_pain",
    "high temperature": "high_fever",
    "temperature": "high_fever",
    "throwing up": "vomiting",
    "puke": "vomiting",
    "tired": "fatigue",
    "tiredness": "tiredness",
    "dizzy": "dizziness",
    "rash": "skin_rash",
    "itch": "itching",
    "itchy": "itching",
    "coughing": "cough",
    "sneeze": "continuous_sneezing",
    "sneezing": "continuous_sneezing",
    "runny nose": "runny_nose",
    "sore throat": "throat_irritation",
    "weight gain": "weight_gain",
    "weight loss": "weight_loss",
    "joint pain": "joint_pain",
    "back pain": "back_pain",
    "neck pain": "neck_pain",
    "chest pain": "chest_pain",
    "stomach pain": "stomach_pain",
    "muscle pain": "muscle_pain",
    "knee pain": "knee_pain",
    "head ache": "headache",
    "head pain": "headache",
    "breathless": "breathlessness",
    "short breath": "breathlessness",
    "constipated": "constipation",
    "diarrhea": "diarrhoea",
    "loose motion": "diarrhoea",
    "nauseous": "nausea",
    "high fever": "high_fever",
    "mild fever": "mild_fever",
    "skin rash": "skin_rash",
    "loss of appetite": "loss_of_appetite",
    "blurred vision": "blurred_and_distorted_vision",
}


def preprocess_query(user_input: str) -> str:
    """
    Preprocess user symptom input for retrieval.
    - Convert to lowercase
    - Remove special characters (keep spaces and underscores)
    - Normalize known symptom aliases
    - Format for embedding
    
    Args:
        user_input: Raw user input string (e.g., "fever headache joint pain")
        
    Returns:
        Preprocessed query string suitable for embedding
    """
    # Lowercase
    query = user_input.lower().strip()
    
    # Remove special characters except spaces, underscores, and commas
    query = re.sub(r'[^a-z0-9\s_,]', '', query)
    
    # Split by comma or space-based delimiters
    # Users might input "fever, headache, joint pain" or "fever headache joint_pain"
    tokens = re.split(r'[,]+', query)
    tokens = [t.strip() for t in tokens if t.strip()]
    
    # If no commas were found, try to identify multi-word symptoms
    if len(tokens) == 1:
        # Check for known multi-word symptom phrases
        remaining = tokens[0]
        identified = []
        
        # Sort aliases by length (longest first) to match multi-word phrases first
        sorted_aliases = sorted(SYMPTOM_ALIASES.keys(), key=len, reverse=True)
        
        for alias in sorted_aliases:
            if alias in remaining:
                identified.append(SYMPTOM_ALIASES[alias])
                remaining = remaining.replace(alias, ' ').strip()
        
        # Any remaining words are treated as individual symptoms
        remaining_words = remaining.split()
        for word in remaining_words:
            word = word.strip()
            if word:
                # Check if it's an alias
                if word in SYMPTOM_ALIASES:
                    identified.append(SYMPTOM_ALIASES[word])
                else:
                    # Replace spaces with underscores for dataset format
                    identified.append(word.replace(' ', '_'))
        
        if identified:
            tokens = identified
        else:
            tokens = [query]
    else:
        # Process each comma-separated token
        processed_tokens = []
        for token in tokens:
            token = token.strip()
            if token in SYMPTOM_ALIASES:
                processed_tokens.append(SYMPTOM_ALIASES[token])
            else:
                processed_tokens.append(token.replace(' ', '_'))
            tokens = processed_tokens
    
    # Build query string: combine symptoms into a natural sentence for embedding
    symptom_text = ", ".join([s.replace("_", " ") for s in tokens])
    query_string = f"Symptoms: {symptom_text}"
    
    return query_string


def extract_symptoms_from_query(user_input: str) -> List[str]:
    """
    Extract individual symptom names from user input for overlap scoring.
    
    Args:
        user_input: Raw user input string
        
    Returns:
        List of normalized symptom names
    """
    query = user_input.lower().strip()
    query = re.sub(r'[^a-z0-9\s_,]', '', query)
    
    # Split by commas first, then spaces
    if ',' in query:
        tokens = [t.strip() for t in query.split(',') if t.strip()]
    else:
        tokens = query.split()
    
    symptoms = []
    i = 0
    while i < len(tokens):
        # Try to match multi-word symptom names
        matched = False
        for length in range(min(4, len(tokens) - i), 0, -1):
            phrase = ' '.join(tokens[i:i + length])
            if phrase in SYMPTOM_ALIASES:
                symptoms.append(SYMPTOM_ALIASES[phrase])
                i += length
                matched = True
                break
        
        if not matched:
            token = tokens[i].replace(' ', '_')
            if token in SYMPTOM_ALIASES:
                symptoms.append(SYMPTOM_ALIASES[token])
            else:
                symptoms.append(token)
            i += 1
    
    return symptoms


def calculate_symptom_overlap(query_symptoms: List[str], doc_symptoms: List[str]) -> float:
    """
    Calculate the overlap score between query symptoms and document symptoms.
    Uses Jaccard-like scoring but weighted towards recall (proportion of
    query symptoms found in the document).
    
    Args:
        query_symptoms: Symptoms extracted from user query
        doc_symptoms: Symptoms associated with a disease document
        
    Returns:
        Overlap score between 0 and 1
    """
    if not query_symptoms or not doc_symptoms:
        return 0.0
    
    query_set = set(query_symptoms)
    doc_set = set(doc_symptoms)
    
    # Count matches (exact and partial)
    matches = 0
    for qs in query_set:
        # Exact match
        if qs in doc_set:
            matches += 1
            continue
        # Partial match (substring matching)
        for ds in doc_set:
            if qs in ds or ds in qs:
                matches += 0.5
                break
    
    # Score = proportion of query symptoms found in document
    score = matches / len(query_set) if query_set else 0.0
    return min(score, 1.0)


def rerank_results(
    results: List[Tuple[Dict, float]],
    query_symptoms: List[str],
    top_n: int = 3,
    vector_weight: float = 0.5,
    overlap_weight: float = 0.35,
    severity_weight: float = 0.15,
) -> List[Dict]:
    """
    Re-rank retrieved results using a combination of:
    1. Vector similarity score (from FAISS)
    2. Symptom overlap score (keyword matching)
    3. Severity score (normalized)
    
    Args:
        results: List of (document, similarity_score) from FAISS search
        query_symptoms: Symptoms extracted from user query
        top_n: Number of top results to return
        vector_weight: Weight for vector similarity (0-1)
        overlap_weight: Weight for symptom overlap (0-1)
        severity_weight: Weight for severity scoring (0-1)
        
    Returns:
        List of result dicts with combined scores, sorted by score descending
    """
    if not results:
        return []
    
    # Find max severity for normalization
    max_severity = max(doc.get('severity_score', 1) for doc, _ in results) or 1
    
    scored_results = []
    for doc, vector_score in results:
        # Normalize vector score to 0-1 range (FAISS IP scores can be < 0)
        normalized_vector = max(0, min(1, (vector_score + 1) / 2))
        
        # Calculate symptom overlap
        doc_symptoms = doc.get('symptoms', [])
        overlap_score = calculate_symptom_overlap(query_symptoms, doc_symptoms)
        
        # Normalize severity
        severity_score = doc.get('severity_score', 0) / max_severity if max_severity > 0 else 0
        
        # Combined score
        combined_score = (
            vector_weight * normalized_vector +
            overlap_weight * overlap_score +
            severity_weight * severity_score
        )
        
        scored_results.append({
            'disease': doc['disease'],
            'confidence': round(combined_score, 4),
            'vector_score': round(normalized_vector, 4),
            'overlap_score': round(overlap_score, 4),
            'severity_score': doc.get('severity_score', 0),
            'symptoms': doc.get('symptoms', []),
            'description': doc.get('description', ''),
            'precautions': doc.get('precautions', []),
            'text': doc.get('text', ''),
        })
    
    # Sort by combined score, descending
    scored_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return scored_results[:top_n]


class DiseaseRetriever:
    """
    High-level retriever that orchestrates query preprocessing, 
    FAISS search, and re-ranking for disease prediction.
    """
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        index_dir: str = "faiss_index"
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Pre-loaded EmbeddingModel (created if None)
            vector_store: Pre-loaded VectorStore (loaded from disk if None)
            index_dir: Directory containing the FAISS index files
        """
        # Load embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = EmbeddingModel()
        
        # Load vector store
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore()
            if not self.vector_store.load(index_dir):
                print("  ⚠ No existing index found. Build one first using vector_store.py")
    
    def retrieve(
        self,
        user_input: str,
        top_k: int = 5,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Retrieve and rank diseases based on user symptom input.
        
        Args:
            user_input: Raw symptom text from user
            top_k: Number of candidates to retrieve from FAISS
            top_n: Number of final results after re-ranking
            
        Returns:
            List of ranked disease result dicts with confidence scores
        """
        # Preprocess query for embedding
        query_string = preprocess_query(user_input)
        
        # Extract symptoms for overlap scoring
        query_symptoms = extract_symptoms_from_query(user_input)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_query(query_string)
        
        # Search FAISS index
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Re-rank results
        ranked_results = rerank_results(raw_results, query_symptoms, top_n=top_n)
        
        return ranked_results


if __name__ == "__main__":
    from src.preprocessing import create_disease_documents
    from src.embeddings import generate_document_embeddings
    from src.vector_store import build_and_save_vector_store
    
    # Build the full pipeline for testing
    print("=" * 60)
    print("Building retrieval pipeline...")
    print("=" * 60)
    
    # Step 1: Create documents
    documents = create_disease_documents()
    
    # Step 2: Generate embeddings
    embeddings = generate_document_embeddings(documents)
    
    # Step 3: Build vector store
    store = build_and_save_vector_store(documents, embeddings)
    
    # Step 4: Create retriever
    model = EmbeddingModel()
    retriever = DiseaseRetriever(embedding_model=model, vector_store=store)
    
    # Test queries
    test_queries = [
        "fever headache joint pain",
        "itching skin rash nodal skin eruptions",
        "burning micturition bladder discomfort",
        "vomiting breathlessness chest pain",
        "cough high fever breathlessness",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")
        
        results = retriever.retrieve(query, top_k=5, top_n=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['disease']} (Confidence: {result['confidence']:.2f})")
            print(f"     Vector: {result['vector_score']:.2f}, "
                  f"Overlap: {result['overlap_score']:.2f}, "
                  f"Severity: {result['severity_score']}")
