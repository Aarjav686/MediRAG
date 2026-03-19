"""
MediRAG - Retrieval Pipeline Tests
Validates that the retrieval pipeline correctly identifies diseases
from known symptom inputs.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import create_disease_documents
from src.embeddings import EmbeddingModel, generate_document_embeddings
from src.vector_store import VectorStore, build_and_save_vector_store
from src.retriever import (
    DiseaseRetriever,
    preprocess_query,
    extract_symptoms_from_query,
    calculate_symptom_overlap,
)


def setup_retriever():
    """Build the full retrieval pipeline for testing."""
    print("Setting up retrieval pipeline for tests...")
    documents = create_disease_documents()
    embeddings = generate_document_embeddings(documents)
    store = build_and_save_vector_store(documents, embeddings)
    model = EmbeddingModel()
    retriever = DiseaseRetriever(embedding_model=model, vector_store=store)
    return retriever


def test_query_preprocessing():
    """Test that query preprocessing normalizes symptoms correctly."""
    print("\n--- Test: Query Preprocessing ---")
    
    # Test basic preprocessing
    result = preprocess_query("fever headache joint pain")
    assert "fever" in result.lower(), f"Expected 'fever' in result: {result}"
    assert "headache" in result.lower(), f"Expected 'headache' in result: {result}"
    print("  ✓ Basic preprocessing works")
    
    # Test alias normalization
    result = preprocess_query("high temperature, coughing, tired")
    assert "fever" in result.lower() or "temperature" in result.lower(), \
        f"Expected fever-related term in result: {result}"
    print("  ✓ Alias normalization works")
    
    # Test symptom extraction
    symptoms = extract_symptoms_from_query("fever headache joint pain")
    assert len(symptoms) >= 3, f"Expected at least 3 symptoms, got {len(symptoms)}: {symptoms}"
    print(f"  ✓ Extracted symptoms: {symptoms}")
    
    print("  ✓ All preprocessing tests passed!")


def test_symptom_overlap():
    """Test symptom overlap scoring."""
    print("\n--- Test: Symptom Overlap Scoring ---")
    
    # Perfect match
    score = calculate_symptom_overlap(
        ["fever", "headache"],
        ["fever", "headache", "joint_pain"]
    )
    assert score > 0.5, f"Expected overlap > 0.5, got {score}"
    print(f"  ✓ Perfect subset match: {score:.2f}")
    
    # No match
    score = calculate_symptom_overlap(
        ["xyz_symptom"],
        ["fever", "headache"]
    )
    assert score == 0.0, f"Expected overlap = 0, got {score}"
    print(f"  ✓ No match: {score:.2f}")
    
    # Empty inputs
    score = calculate_symptom_overlap([], ["fever"])
    assert score == 0.0
    print(f"  ✓ Empty query: {score:.2f}")
    
    print("  ✓ All overlap tests passed!")


def test_dengue_retrieval(retriever):
    """Test that dengue-related symptoms retrieve Dengue."""
    print("\n--- Test: Dengue Retrieval ---")
    results = retriever.retrieve("fever headache joint pain", top_k=5, top_n=3)
    
    diseases = [r['disease'] for r in results]
    assert "Dengue" in diseases, f"Expected 'Dengue' in top-3, got: {diseases}"
    
    # Verify confidence scores are in valid range
    for r in results:
        assert 0 <= r['confidence'] <= 1, \
            f"Confidence out of range: {r['confidence']} for {r['disease']}"
    
    print(f"  ✓ Top diseases: {diseases}")
    print(f"  ✓ Dengue found with confidence: {results[diseases.index('Dengue')]['confidence']:.2f}")
    print("  ✓ Dengue retrieval test passed!")


def test_fungal_infection_retrieval(retriever):
    """Test that skin-related symptoms retrieve Fungal infection."""
    print("\n--- Test: Fungal Infection Retrieval ---")
    results = retriever.retrieve("itching skin rash nodal skin eruptions", top_k=5, top_n=3)
    
    diseases = [r['disease'] for r in results]
    assert "Fungal infection" in diseases, \
        f"Expected 'Fungal infection' in top-3, got: {diseases}"
    
    print(f"  ✓ Top diseases: {diseases}")
    print("  ✓ Fungal infection retrieval test passed!")


def test_diabetes_retrieval(retriever):
    """Test that diabetes-related symptoms retrieve Diabetes."""
    print("\n--- Test: Diabetes Retrieval ---")
    results = retriever.retrieve("fatigue weight loss blurred vision excessive hunger", top_k=5, top_n=3)
    
    diseases = [r['disease'].strip() for r in results]
    # Note: disease name has trailing space in dataset: "Diabetes "
    has_diabetes = any("Diabetes" in d for d in diseases)
    assert has_diabetes, f"Expected 'Diabetes' in top-3, got: {diseases}"
    
    print(f"  ✓ Top diseases: {diseases}")
    print("  ✓ Diabetes retrieval test passed!")


def test_score_normalization(retriever):
    """Test that all confidence scores are normalized between 0 and 1."""
    print("\n--- Test: Score Normalization ---")
    
    test_queries = [
        "fever headache",
        "cough breathlessness chest pain",
        "vomiting diarrhoea dehydration",
    ]
    
    for query in test_queries:
        results = retriever.retrieve(query, top_k=5, top_n=3)
        for r in results:
            assert 0 <= r['confidence'] <= 1, \
                f"Score out of range for '{query}': {r['disease']}={r['confidence']}"
            assert 0 <= r['vector_score'] <= 1, \
                f"Vector score out of range: {r['disease']}={r['vector_score']}"
            assert 0 <= r['overlap_score'] <= 1, \
                f"Overlap score out of range: {r['disease']}={r['overlap_score']}"
    
    print("  ✓ All scores properly normalized (0-1)")
    print("  ✓ Score normalization test passed!")


def test_result_structure(retriever):
    """Test that results contain all expected fields."""
    print("\n--- Test: Result Structure ---")
    results = retriever.retrieve("fever cough", top_k=5, top_n=3)
    
    required_fields = [
        'disease', 'confidence', 'vector_score', 'overlap_score',
        'severity_score', 'symptoms', 'description', 'precautions', 'text'
    ]
    
    for r in results:
        for field in required_fields:
            assert field in r, f"Missing field '{field}' in result for {r.get('disease', 'unknown')}"
    
    assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"
    assert len(results) > 0, "Expected at least 1 result"
    
    print(f"  ✓ All {len(required_fields)} required fields present in {len(results)} results")
    print("  ✓ Result structure test passed!")


def run_all_tests():
    """Run all retrieval pipeline tests."""
    print("=" * 60)
    print("MediRAG - Retrieval Pipeline Test Suite")
    print("=" * 60)
    
    # Tests that don't need the full pipeline
    test_query_preprocessing()
    test_symptom_overlap()
    
    # Build the retriever
    retriever = setup_retriever()
    
    # Full pipeline tests
    test_dengue_retrieval(retriever)
    test_fungal_infection_retrieval(retriever)
    test_diabetes_retrieval(retriever)
    test_score_normalization(retriever)
    test_result_structure(retriever)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
