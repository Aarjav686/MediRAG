import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from src.preprocessing import get_all_symptoms
from src.llm import MEDICAL_DISCLAIMER

# Configure Streamlit page
st.set_page_config(
    page_title="Symptom-to-Disease AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and risk level colors
st.markdown("""
<style>
    .risk-Low { color: #2e7d32; font-weight: bold; }
    .risk-Moderate { color: #f57c00; font-weight: bold; }
    .risk-High { color: #d32f2f; font-weight: bold; }
    .risk-Critical { color: #b71c1c; font-weight: bold; text-decoration: underline; }
    
    .disease-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    
    .disclaimer-footer {
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #666;
        border-top: 1px solid #eee;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Caching the Pipeline
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI Models (this may take a moment)...")
def load_medirag_pipeline(use_llm: bool = False):
    """
    Load the RAG pipeline. Cached so models don't reload on every interaction.
    """
    return RAGPipeline(use_llm=use_llm)

@st.cache_data
def load_symptoms_list():
    """Load the list of valid symptoms from the dataset."""
    return get_all_symptoms()

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("MediRAG Settings")
    
    st.markdown("### Settings")
    use_llm_toggle = st.checkbox(
        "Enable TinyLlama LLM", 
        value=False,
        help="If checked, uses the AI model for explanations. Otherwise, uses a faster template."
    )
    
    st.markdown("---")
    
    # Load pipeline based on toggle
    pipeline = load_medirag_pipeline(use_llm=use_llm_toggle)
    system_info = pipeline.get_system_info()
    
    st.markdown("### System Status")
    st.write(f"- **Mode:** {system_info['llm']['mode']}")
    st.write(f"- **Documents Index:** {system_info['retriever']['num_documents']} diseases")
    
    st.markdown("---")
    
    st.markdown("### Common Symptoms")
    st.markdown("Use these terms for better results:")
    
    symptoms_list = load_symptoms_list()
    if symptoms_list:
        with st.expander("View Supported Symptoms"):
            # Show a sample or formatted list
            formatted_list = ", ".join([s.replace("_", " ") for s in symptoms_list])
            st.write(formatted_list)

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
st.title("🏥 Symptom-to-Disease AI Assistant")
st.markdown("""
Welcome to **MediRAG**, an AI-powered medical knowledge assistant. 
Please describe your symptoms below, separated by commas (e.g., _"high fever, severe headache, joint pain"_).
""")

# User input
user_input = st.text_area("Describe your symptoms:", height=100, 
                          placeholder="e.g., fever, headache, joint pain, muscle pain")

analyze_button = st.button("Analyze Symptoms", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Analysis and Results
# -----------------------------------------------------------------------------
if analyze_button:
    if not user_input.strip():
        st.warning("Please enter some symptoms before analyzing.")
    else:
        with st.spinner("Analyzing symptoms and searching medical knowledge base..."):
            # Run the pipeline
            result = pipeline.analyze_symptoms(user_input)
            
            predictions = result.get("predictions", [])
            risk_level = result.get("risk_level", "Unknown")
            explanation = result.get("explanation", "")
            
            st.markdown("---")
            st.header("Analysis Results")
            
            # 1. Overall Risk Level
            risk_color_map = {
                "Low": "success",
                "Moderate": "warning",
                "High": "error",
                "Critical": "error",
                "Unknown": "info"
            }
            
            # Display risk alert
            if risk_level == "Critical":
                st.error(f"🚨 **Overall Risk Level: CRITICAL** - Immediate medical attention is highly recommended.")
            elif risk_level == "High":
                st.error(f"⚠️ **Overall Risk Level: High** - Please consult a doctor soon.")
            elif risk_level == "Moderate":
                st.warning(f"🔔 **Overall Risk Level: Moderate** - Monitor symptoms and consider a medical checkup.")
            elif risk_level == "Low":
                st.success(f"✅ **Overall Risk Level: Low** - Symptoms indicate a lower risk, but consult a doctor if they persist.")
            
            # 2. AI Explanation
            st.markdown("### AI Summary")
            st.info(explanation)
            
            # 3. Top Predictions
            if predictions:
                st.markdown("### Top Disease Predictions")
                
                # Create tabs for the top predictions
                tabs = st.tabs([f"#{i+1}: {p['disease']}" for i, p in enumerate(predictions)])
                
                for i, (tab, pred) in enumerate(zip(tabs, predictions)):
                    with tab:
                        st.markdown(f"<div class='disease-card'>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader(pred['disease'])
                            st.write(pred.get('description', 'No description available.'))
                        
                        with col2:
                            confidence = pred.get('confidence', 0)
                            st.metric("Confidence Score", f"{confidence:.0%}")
                            st.progress(confidence)
                            
                            p_risk = pred.get('risk_level', 'Unknown')
                            st.markdown(f"**Risk Level:** <span class='risk-{p_risk}'>{p_risk}</span>", unsafe_allow_html=True)
                        
                        st.markdown("#### Matched Symptoms:")
                        matched_symptoms = pred.get('symptoms', [])
                        if matched_symptoms:
                            # Format nicely
                            symp_text = ", ".join([s.replace("_", " ").title() for s in matched_symptoms[:10]])
                            st.write(symp_text)
                        
                        st.markdown("#### Recommended Precautions:")
                        precautions = pred.get('precautions', [])
                        if precautions:
                            for p in precautions:
                                st.write(f"- {p.capitalize()}")
                        else:
                            st.write("No specific precautions listed.")
                            
                        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Footer / Disclaimer
# -----------------------------------------------------------------------------
st.markdown(f"<div class='disclaimer-footer'>{MEDICAL_DISCLAIMER}</div>", unsafe_allow_html=True)
