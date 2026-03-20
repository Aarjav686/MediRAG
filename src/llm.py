"""
MediRAG - LLM Module
Loads and configures a local LLM (TinyLlama-1.1B-Chat) for generating
human-readable medical explanations from retrieved disease context.
Includes a template-based fallback for systems without enough memory.
"""

import traceback
from typing import List, Dict, Optional


# Medical disclaimer included in all outputs
MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This is an AI-powered educational tool only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider for medical concerns."
)

# Default LLM model
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# TinyLlama chat template tokens
_EOS_TOKEN = "</s>"
_SYS_TOKEN = "<|system|>"
_USR_TOKEN = "<|user|>"
_AST_TOKEN = "<|assistant|>"


def classify_risk_level(severity_score: float) -> str:
    """
    Classify disease risk level based on average symptom severity score.

    Args:
        severity_score: Average severity score from preprocessing (0-7 scale)

    Returns:
        Risk level string: Low, Moderate, High, or Critical
    """
    if severity_score < 3:
        return "Low"
    elif severity_score < 4.5:
        return "Moderate"
    elif severity_score < 5.5:
        return "High"
    else:
        return "Critical"


def _build_chat_prompt(context_docs: List[Dict], symptoms: str) -> str:
    """
    Build a TinyLlama chat-format prompt for medical explanation generation.

    Args:
        context_docs: List of retrieved disease documents with scores
        symptoms: User's original symptom input

    Returns:
        Formatted prompt string in TinyLlama chat template
    """
    # Build context block from retrieved documents
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        risk = classify_risk_level(doc.get("severity_score", 0))
        confidence = doc.get("confidence", 0)
        context_parts.append(
            f"Disease {i}: {doc['disease']}\n"
            f"  Confidence: {confidence:.0%}\n"
            f"  Risk Level: {risk}\n"
            f"  Description: {doc.get('description', 'N/A')}\n"
            f"  Key Symptoms: {', '.join(doc.get('symptoms', [])[:8])}\n"
            f"  Precautions: {', '.join(doc.get('precautions', []))}"
        )
    context_block = "\n\n".join(context_parts)

    system_msg = (
        "You are a medical knowledge assistant. Based on the retrieved disease "
        "information below, provide a clear, concise explanation of the most "
        "likely conditions matching the patient's symptoms. "
        "For each condition: briefly explain why the symptoms match, "
        "list the key precautions, and state the risk level. "
        "Keep the response under 300 words. "
        "End with a reminder to consult a doctor."
    )

    user_msg = (
        f"Patient reports these symptoms: {symptoms}\n\n"
        f"Retrieved medical information:\n{context_block}\n\n"
        "Please provide a brief medical analysis."
    )

    # TinyLlama chat template format
    prompt = (
        _SYS_TOKEN + "\n"
        + system_msg + _EOS_TOKEN + "\n"
        + _USR_TOKEN + "\n"
        + user_msg + _EOS_TOKEN + "\n"
        + _AST_TOKEN + "\n"
    )
    return prompt


def template_fallback(context_docs: List[Dict], symptoms: str) -> str:
    """
    Generate a structured template-based explanation when the LLM is unavailable.
    Falls back to formatting the retrieved context directly.

    Args:
        context_docs: List of retrieved disease documents with scores
        symptoms: User's original symptom input

    Returns:
        Formatted explanation string
    """
    if not context_docs:
        return (
            "No matching conditions were found for the reported symptoms. "
            "Please try describing your symptoms differently or consult a healthcare provider."
        )

    lines = []
    lines.append(f"Based on the symptoms reported ({symptoms}), "
                 f"here are the most likely conditions:\n")

    for i, doc in enumerate(context_docs, 1):
        disease = doc.get("disease", "Unknown")
        confidence = doc.get("confidence", 0)
        risk = classify_risk_level(doc.get("severity_score", 0))
        description = doc.get("description", "No description available.")
        precautions = doc.get("precautions", [])
        matching_symptoms = doc.get("symptoms", [])

        lines.append(f"--- {i}. {disease} (Confidence: {confidence:.0%}, Risk: {risk}) ---")
        lines.append(f"{description[:200]}")

        if matching_symptoms:
            symptom_text = ", ".join([s.replace("_", " ") for s in matching_symptoms[:6]])
            lines.append(f"Key symptoms: {symptom_text}")

        if precautions:
            lines.append("Recommended precautions:")
            for p in precautions:
                lines.append(f"  - {p}")

        lines.append("")

    lines.append(MEDICAL_DISCLAIMER)
    return "\n".join(lines)


class MediLLM:
    """
    Local LLM wrapper for generating medical explanations.
    Uses TinyLlama-1.1B-Chat by default with a template-based fallback.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, use_llm: bool = True):
        """
        Initialize the LLM.

        Args:
            model_name: HuggingFace model identifier
            use_llm: If False, skip model loading and always use template fallback
        """
        self.model_name = model_name
        self.use_llm = use_llm
        self.pipeline = None
        self._load_failed = False

        if use_llm:
            self._load_model()

    def _load_model(self):
        """Load the TinyLlama model via HuggingFace transformers pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            print(f"Loading LLM: {self.model_name}...")

            # Detect available device
            if torch.cuda.is_available():
                device = 0  # First GPU
                dtype = torch.float16
                print("  Using CUDA GPU")
            else:
                device = -1  # CPU
                dtype = torch.float32
                print("  Using CPU (this may be slow)")

            self.pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=dtype,
                device=device,
            )
            print("  Model loaded successfully!")

        except Exception as e:
            print(f"  Failed to load LLM: {e}")
            print("  Falling back to template-based explanations.")
            self._load_failed = True
            self.pipeline = None

    def generate_explanation(
        self,
        context_docs: List[Dict],
        symptoms: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a medical explanation from retrieved disease context.

        If the LLM is available, generates using TinyLlama.
        Otherwise, falls back to a structured template.

        Args:
            context_docs: List of retrieved disease documents
            symptoms: User's original symptom input
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated explanation string
        """
        # Use template fallback if LLM is not available
        if not self.use_llm or self.pipeline is None:
            return template_fallback(context_docs, symptoms)

        try:
            # Build the chat prompt
            prompt = _build_chat_prompt(context_docs, symptoms)

            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                return_full_text=False,
            )

            response = outputs[0]["generated_text"].strip()

            # Post-process: clean up incomplete sentences
            response = _postprocess_response(response)

            # Ensure disclaimer is present
            if "disclaimer" not in response.lower() and "consult" not in response.lower():
                response += "\n\n" + MEDICAL_DISCLAIMER

            return response

        except Exception as e:
            print(f"  LLM generation failed: {e}")
            traceback.print_exc()
            return template_fallback(context_docs, symptoms)


def _postprocess_response(response: str) -> str:
    """
    Clean up LLM-generated response.
    - Remove incomplete trailing sentences
    - Strip excessive whitespace

    Args:
        response: Raw LLM output string

    Returns:
        Cleaned response string
    """
    if not response:
        return ""

    # Strip whitespace
    response = response.strip()

    # If response ends mid-sentence, cut at the last complete sentence
    if response and response[-1] not in '.!?':
        last_period = response.rfind('.')
        last_excl = response.rfind('!')
        last_q = response.rfind('?')
        last_end = max(last_period, last_excl, last_q)
        if last_end > len(response) // 2:  # Only cut if we keep at least half
            response = response[:last_end + 1]

    return response


if __name__ == "__main__":
    # Quick test with template fallback
    print("=" * 60)
    print("MediRAG LLM Module - Template Fallback Test")
    print("=" * 60)

    # Test with sample context
    sample_docs = [
        {
            'disease': 'Dengue',
            'confidence': 0.85,
            'severity_score': 4.2,
            'description': 'Dengue fever is a mosquito-borne tropical disease caused by the dengue virus.',
            'symptoms': ['fever', 'headache', 'joint_pain', 'muscle_pain', 'skin_rash'],
            'precautions': ['drink papaya leaf juice', 'avoid fatty food', 'keep hydrated', 'keep mosquitos away'],
        },
        {
            'disease': 'Malaria',
            'confidence': 0.62,
            'severity_score': 5.1,
            'description': 'Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites.',
            'symptoms': ['fever', 'headache', 'chills', 'sweating', 'nausea'],
            'precautions': ['Consult nearest hospital', 'avoid oily food', 'avoid non veg food', 'keep mosquitos out'],
        },
    ]

    # Test template fallback
    llm = MediLLM(use_llm=False)
    explanation = llm.generate_explanation(sample_docs, "fever headache joint pain")
    print(explanation)

    print("\n" + "=" * 60)
    print("Risk Level Classification:")
    for score in [1.5, 3.0, 4.5, 5.5, 6.5]:
        print(f"  Severity {score} -> {classify_risk_level(score)}")
