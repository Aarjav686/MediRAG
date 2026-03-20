"""
MediRAG - RAG Pipeline Module
Orchestrates the full Retrieval-Augmented Generation pipeline:
preprocessing -> retrieval -> context building -> LLM generation.
"""

from typing import List, Dict, Optional
from src.retriever import DiseaseRetriever
from src.llm import MediLLM, classify_risk_level, MEDICAL_DISCLAIMER


class RAGPipeline:
    """
    Full RAG pipeline for symptom-to-disease analysis.
    Combines FAISS retrieval with LLM-based explanation generation.
    """

    def __init__(
        self,
        index_dir: str = "faiss_index",
        use_llm: bool = True,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            index_dir: Directory containing the FAISS index
            use_llm: Whether to use TinyLlama (False = template fallback only)
            model_name: HuggingFace model identifier for the LLM
        """
        print("=" * 60)
        print("Initializing MediRAG Pipeline...")
        print("=" * 60)

        # Initialize retriever
        print("\n[1/2] Loading retriever...")
        self.retriever = DiseaseRetriever(index_dir=index_dir)

        # Initialize LLM
        print("\n[2/2] Loading LLM...")
        self.llm = MediLLM(model_name=model_name, use_llm=use_llm)

        self.use_llm = use_llm
        print("\n" + "=" * 60)
        print("MediRAG Pipeline ready!")
        print("=" * 60)

    def analyze_symptoms(
        self,
        user_input: str,
        top_k: int = 5,
        top_n: int = 3,
    ) -> Dict:
        """
        Analyze user symptoms through the full RAG pipeline.

        Flow:
        1. Retrieve top disease matches via FAISS + re-ranking
        2. Classify risk level for each prediction
        3. Generate explanation via LLM (or template fallback)
        4. Return structured results with disclaimer

        Args:
            user_input: Raw symptom text from user (e.g., "fever headache joint pain")
            top_k: Number of FAISS candidates to retrieve
            top_n: Number of final predictions after re-ranking

        Returns:
            Dict with keys:
                - predictions: List of disease prediction dicts
                - explanation: LLM-generated or template explanation string
                - risk_level: Overall risk level (highest among predictions)
                - disclaimer: Medical disclaimer string
        """
        if not user_input or not user_input.strip():
            return {
                "predictions": [],
                "explanation": "Please enter your symptoms to get an analysis.",
                "risk_level": "Unknown",
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        # Step 1: Retrieve and rank diseases
        predictions = self.retriever.retrieve(
            user_input, top_k=top_k, top_n=top_n
        )

        # Step 2: Add risk level to each prediction
        for pred in predictions:
            pred["risk_level"] = classify_risk_level(
                pred.get("severity_score", 0)
            )

        # Step 3: Determine overall risk level (highest among predictions)
        if predictions:
            risk_order = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
            overall_risk = max(
                predictions,
                key=lambda p: risk_order.get(p.get("risk_level", "Low"), 0),
            )["risk_level"]
        else:
            overall_risk = "Unknown"

        # Step 4: Generate explanation
        explanation = self.llm.generate_explanation(predictions, user_input)

        # Step 5: Handle edge case — all low confidence
        if predictions and all(p.get("confidence", 0) < 0.3 for p in predictions):
            explanation = (
                "The confidence scores for all matched conditions are low. "
                "This may mean your symptoms don't closely match common disease patterns "
                "in our knowledge base. Please consult a healthcare professional "
                "for a proper evaluation.\n\n" + explanation
            )

        return {
            "predictions": predictions,
            "explanation": explanation,
            "risk_level": overall_risk,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def get_system_info(self) -> Dict:
        """
        Get information about the loaded pipeline components.

        Returns:
            Dict with system status information
        """
        info = {
            "retriever": {
                "index_loaded": self.retriever.vector_store.index is not None,
                "num_documents": len(self.retriever.vector_store.documents),
                "embedding_dim": self.retriever.vector_store.embedding_dim,
            },
            "llm": {
                "model_name": self.llm.model_name,
                "use_llm": self.llm.use_llm,
                "model_loaded": self.llm.pipeline is not None,
                "mode": "LLM" if (self.llm.use_llm and self.llm.pipeline) else "Template Fallback",
            },
        }
        return info


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("MediRAG - RAG Pipeline Smoke Test")
    print("=" * 60)

    # Initialize pipeline with template fallback (no LLM download needed)
    pipeline = RAGPipeline(use_llm=False)

    # Print system info
    print("\nSystem Info:")
    info = pipeline.get_system_info()
    print(json.dumps(info, indent=2))

    # Test queries
    test_queries = [
        "fever headache joint pain",
        "itching skin rash",
        "burning micturition bladder discomfort",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")

        result = pipeline.analyze_symptoms(query)

        print(f"\nOverall Risk Level: {result['risk_level']}")
        print(f"\nPredictions:")
        for i, pred in enumerate(result["predictions"], 1):
            print(
                f"  {i}. {pred['disease']} "
                f"(Confidence: {pred['confidence']:.0%}, "
                f"Risk: {pred['risk_level']})"
            )

        print(f"\nExplanation (first 300 chars):")
        print(f"  {result['explanation'][:300]}...")

    # Test empty input
    print(f"\n{'=' * 60}")
    print("Test: Empty input")
    result = pipeline.analyze_symptoms("")
    print(f"  Result: {result['explanation']}")
