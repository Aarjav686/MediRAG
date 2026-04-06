"""
MediRAG - Flask Web Application
Replaces the Streamlit UI with a premium HTML/CSS/JS interface.
"""

import os
import sys
import csv
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from src.preprocessing import get_all_symptoms
from src.llm import MEDICAL_DISCLAIMER

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# ---------------------------------------------------------------------------
# Pipeline singleton (lazy-loaded once)
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline(use_llm: bool = False) -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(use_llm=use_llm)
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/symptoms", methods=["GET"])
def api_symptoms():
    """Return the list of all supported symptoms."""
    try:
        symptoms = get_all_symptoms()
        formatted = [s.replace("_", " ") for s in symptoms]
        return jsonify({"symptoms": formatted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def api_status():
    """Return system status information."""
    try:
        pipeline = get_pipeline()
        info = pipeline.get_system_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze symptoms and return disease predictions."""
    data = request.get_json(force=True)
    user_input = data.get("symptoms", "").strip()
    use_llm = data.get("use_llm", False)

    if not user_input:
        return jsonify({"error": "Please enter some symptoms."}), 400

    try:
        pipeline = get_pipeline(use_llm=use_llm)
        result = pipeline.analyze_symptoms(user_input)

        predictions = result.get("predictions", [])
        risk_level = result.get("risk_level", "Unknown")
        explanation = result.get("explanation", "")

        # Log query
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "query_log.csv")
        log_exists = os.path.exists(log_file)
        top_disease = predictions[0]["disease"] if predictions else "Unknown"
        top_confidence = predictions[0]["confidence"] if predictions else 0.0

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(["Timestamp", "User Input", "Top Prediction", "Confidence", "Risk Level"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_input,
                top_disease,
                f"{top_confidence:.4f}",
                risk_level,
            ])

        return jsonify({
            "predictions": predictions,
            "risk_level": risk_level,
            "explanation": explanation,
            "disclaimer": MEDICAL_DISCLAIMER,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------------------------

@app.route("/api/upload-pdf", methods=["POST"])
def api_upload_pdf():
    """
    Extract text from an uploaded PDF.
    Strategy:
      1. Try PyPDF2 text extraction (fast, digital PDFs).
      2. If no text found, render pages with PyMuPDF and run EasyOCR (scanned PDFs).
    Returns extracted text, page count, filename, and extraction method used.
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file.filename or not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file."}), 400

    pdf_bytes = pdf_file.read()
    filename = pdf_file.filename
    num_pages = 0

    # ── Stage 1: PyPDF2 text extraction ─────────────────────────
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text).strip()
        if full_text:
            return jsonify({
                "text": full_text,
                "pages": num_pages,
                "filename": filename,
                "method": "text",          # digital PDF — fast extraction
            })
    except Exception as e:
        print(f"[PDF] PyPDF2 error: {e}")

    # ── Stage 2: OCR via PyMuPDF + EasyOCR ──────────────────────
    try:
        import fitz                        # PyMuPDF
        import easyocr
        import numpy as np

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)

        # Lazy-load EasyOCR reader (cached on app context)
        if not hasattr(app, "_ocr_reader"):
            print("[OCR] Loading EasyOCR model (first time, may take ~30s)…")
            app._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[OCR] EasyOCR ready.")

        reader = app._ocr_reader
        ocr_texts = []

        for page_num in range(num_pages):
            page = doc[page_num]
            # Render at 2× resolution for better OCR accuracy
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # Convert pixmap to numpy RGB array without saving to disk
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            results = reader.readtext(img_array, detail=0, paragraph=True)
            page_text = " ".join(results).strip()
            if page_text:
                ocr_texts.append(page_text)

        doc.close()
        full_text = "\n\n".join(ocr_texts).strip()

        if not full_text:
            return jsonify({
                "error": "OCR could not extract any text from this PDF. "
                         "The document may be image-only or too low quality."
            }), 422

        return jsonify({
            "text": full_text,
            "pages": num_pages,
            "filename": filename,
            "method": "ocr",              # scanned PDF — OCR used
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pre-load pipeline on startup
    print("Pre-loading MediRAG pipeline...")
    get_pipeline(use_llm=False)
    app.run(debug=True, host="0.0.0.0", port=5050)
