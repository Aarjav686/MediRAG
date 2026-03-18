"""
MediRAG - Data Preprocessing Module
Merges all dataset files into a unified knowledge base and creates
document representations for each disease suitable for RAG retrieval.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.data_loader import load_all_data


def extract_symptoms_per_disease(dataset_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extract unique symptom lists for each disease from the dataset.csv file.
    The dataset has columns: Disease, Symptom_0, Symptom_1, ..., Symptom_16
    
    Args:
        dataset_df: DataFrame from dataset.csv
        
    Returns:
        Dictionary mapping disease name -> list of unique symptoms
    """
    disease_symptoms = {}
    
    # Get all symptom columns (everything except 'Disease')
    symptom_cols = [col for col in dataset_df.columns if col != 'Disease']
    
    for disease in dataset_df['Disease'].unique():
        # Get all rows for this disease
        disease_rows = dataset_df[dataset_df['Disease'] == disease]
        
        # Collect all symptoms across all rows
        symptoms = set()
        for _, row in disease_rows.iterrows():
            for col in symptom_cols:
                symptom = str(row[col]).strip()
                # Skip null/NaN values
                if symptom and symptom.lower() != 'nan' and symptom.lower() != 'null':
                    symptoms.add(symptom)
        
        disease_symptoms[disease.strip()] = sorted(list(symptoms))
    
    return disease_symptoms


def build_severity_lookup(severity_df: pd.DataFrame) -> Dict[str, int]:
    """
    Build a dictionary mapping symptom names to their severity scores.
    
    Args:
        severity_df: DataFrame from symptom_severity.csv
        
    Returns:
        Dictionary mapping symptom -> severity score (1-7)
    """
    severity_lookup = {}
    for _, row in severity_df.iterrows():
        symptom = str(row['Symptom']).strip()
        severity = int(row['Symptom_severity'])
        severity_lookup[symptom] = severity
    
    return severity_lookup


def build_description_lookup(descriptions_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a dictionary mapping disease names to their descriptions.
    
    Args:
        descriptions_df: DataFrame from disease_description.csv
        
    Returns:
        Dictionary mapping disease -> description text
    """
    desc_lookup = {}
    for _, row in descriptions_df.iterrows():
        disease = str(row['Disease']).strip()
        description = str(row['Symptom_Description']).strip()
        desc_lookup[disease] = description
    
    return desc_lookup


def build_precaution_lookup(precautions_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a dictionary mapping disease names to their precaution lists.
    
    Args:
        precautions_df: DataFrame from disease_precaution.csv
        
    Returns:
        Dictionary mapping disease -> list of precaution strings
    """
    prec_lookup = {}
    precaution_cols = [col for col in precautions_df.columns if col != 'Disease']
    
    for _, row in precautions_df.iterrows():
        disease = str(row['Disease']).strip()
        precautions = []
        for col in precaution_cols:
            precaution = str(row[col]).strip()
            if precaution and precaution.lower() != 'nan' and precaution.lower() != 'null':
                precautions.append(precaution)
        prec_lookup[disease] = precautions
    
    return prec_lookup


def calculate_disease_severity(symptoms: List[str], severity_lookup: Dict[str, int]) -> float:
    """
    Calculate aggregate severity score for a disease based on its symptoms.
    
    Args:
        symptoms: List of symptom names for a disease
        severity_lookup: Dictionary mapping symptom -> severity score
        
    Returns:
        Average severity score (0 if no symptoms have severity data)
    """
    scores = []
    for symptom in symptoms:
        if symptom in severity_lookup:
            scores.append(severity_lookup[symptom])
    
    return round(np.mean(scores), 2) if scores else 0.0


def create_disease_documents(data_dir: str = "data") -> List[Dict]:
    """
    Create structured document representations for each disease by merging
    all dataset files into a unified knowledge base.
    
    Each document contains:
        - disease: Disease name
        - symptoms: List of symptom names
        - description: Medical description
        - precautions: List of precaution strings
        - severity_score: Average severity of symptoms
        - text: Full text representation for embedding
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        List of document dictionaries
    """
    # Load all data
    data = load_all_data(data_dir)
    
    # Build lookup tables
    disease_symptoms = extract_symptoms_per_disease(data['dataset'])
    severity_lookup = build_severity_lookup(data['severity'])
    desc_lookup = build_description_lookup(data['descriptions'])
    prec_lookup = build_precaution_lookup(data['precautions'])
    
    documents = []
    
    for disease, symptoms in disease_symptoms.items():
        # Get description (with fallback)
        description = desc_lookup.get(disease, "No description available.")
        
        # Get precautions (with fallback)
        precautions = prec_lookup.get(disease, [])
        
        # Calculate severity score
        severity_score = calculate_disease_severity(symptoms, severity_lookup)
        
        # Format symptoms as readable text (replace underscores with spaces)
        symptoms_text = ", ".join([s.replace("_", " ") for s in symptoms])
        precautions_text = ", ".join(precautions) if precautions else "No specific precautions listed."
        
        # Build the full document text for embedding
        doc_text = (
            f"Disease: {disease}\n"
            f"Symptoms: {symptoms_text}\n"
            f"Description: {description}\n"
            f"Precautions: {precautions_text}\n"
            f"Average Severity Score: {severity_score}"
        )
        
        document = {
            'disease': disease,
            'symptoms': symptoms,
            'description': description,
            'precautions': precautions,
            'severity_score': severity_score,
            'text': doc_text,
        }
        
        documents.append(document)
    
    print(f"\nCreated {len(documents)} disease documents for the knowledge base.")
    return documents


def get_all_symptoms(data_dir: str = "data") -> List[str]:
    """
    Get a sorted list of all unique symptom names from the dataset.
    Useful for displaying available symptoms to the user.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Sorted list of unique symptom names
    """
    data = load_all_data(data_dir)
    disease_symptoms = extract_symptoms_per_disease(data['dataset'])
    
    all_symptoms = set()
    for symptoms in disease_symptoms.values():
        all_symptoms.update(symptoms)
    
    return sorted(list(all_symptoms))


if __name__ == "__main__":
    # Quick test: create documents and print sample
    documents = create_disease_documents()
    
    print("\n" + "=" * 60)
    print("SAMPLE DOCUMENT")
    print("=" * 60)
    
    # Print first document
    doc = documents[0]
    print(f"Disease: {doc['disease']}")
    print(f"Symptoms ({len(doc['symptoms'])}): {', '.join(doc['symptoms'][:5])}...")
    print(f"Description: {doc['description'][:100]}...")
    print(f"Precautions: {doc['precautions']}")
    print(f"Severity Score: {doc['severity_score']}")
    print(f"\nFull Text:\n{doc['text'][:300]}...")
    
    print(f"\n--- Total unique symptoms: {len(get_all_symptoms())} ---")
