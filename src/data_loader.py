"""
MediRAG - Data Loader Module
Handles loading of all CSV dataset files for the medical RAG system.
"""

import os
import pandas as pd


def load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the main dataset with disease-symptom mappings.
    Each row has a disease and its associated symptoms as named columns.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        DataFrame with Disease and Symptom columns
    """
    filepath = os.path.join(data_dir, "dataset.csv")
    df = pd.read_csv(filepath)
    return df


def load_training_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the training data with binary symptom-disease matrix.
    Columns are symptom names (0/1 values), last column is 'prognosis' (disease name).
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        DataFrame with binary symptom columns and prognosis column
    """
    filepath = os.path.join(data_dir, "Training.csv")
    df = pd.read_csv(filepath)
    return df


def load_testing_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the testing data with binary symptom-disease matrix.
    Same format as training data but smaller (one row per disease).
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        DataFrame with binary symptom columns and prognosis column
    """
    filepath = os.path.join(data_dir, "Testing.csv")
    df = pd.read_csv(filepath)
    return df


def load_descriptions(data_dir: str = "data") -> pd.DataFrame:
    """
    Load disease descriptions.
    Each row: Disease, Symptom_Description
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        DataFrame with Disease and Symptom_Description columns
    """
    filepath = os.path.join(data_dir, "disease_description.csv")
    df = pd.read_csv(filepath)
    return df


def load_precautions(data_dir: str = "data") -> pd.DataFrame:
    """
    Load disease precautions.
    Each row: Disease, Symptom_precaution_0 through Symptom_precaution_3
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        DataFrame with Disease and 4 precaution columns
    """
    filepath = os.path.join(data_dir, "disease_precaution.csv")
    df = pd.read_csv(filepath)
    return df


def load_severity(data_dir: str = "data") -> pd.DataFrame:
    """
    Load symptom severity scores.
    Each row: Symptom, Symptom_severity (integer 1-7)
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        DataFrame with Symptom and Symptom_severity columns
    """
    filepath = os.path.join(data_dir, "symptom_severity.csv")
    df = pd.read_csv(filepath)
    return df


def load_all_data(data_dir: str = "data") -> dict:
    """
    Load all dataset files and return them as a dictionary.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Dictionary with keys:
            - 'dataset': Main disease-symptom dataset
            - 'training': Binary training matrix
            - 'testing': Binary testing matrix
            - 'descriptions': Disease descriptions
            - 'precautions': Disease precautions
            - 'severity': Symptom severity scores
    """
    print("Loading dataset files...")
    
    data = {
        'dataset': load_dataset(data_dir),
        'training': load_training_data(data_dir),
        'testing': load_testing_data(data_dir),
        'descriptions': load_descriptions(data_dir),
        'precautions': load_precautions(data_dir),
        'severity': load_severity(data_dir),
    }
    
    # Print summary statistics
    print(f"  ✓ Dataset: {len(data['dataset'])} rows, {len(data['dataset']['Disease'].unique())} diseases")
    print(f"  ✓ Training: {len(data['training'])} rows")
    print(f"  ✓ Testing: {len(data['testing'])} rows")
    print(f"  ✓ Descriptions: {len(data['descriptions'])} diseases")
    print(f"  ✓ Precautions: {len(data['precautions'])} diseases")
    print(f"  ✓ Severity: {len(data['severity'])} symptoms")
    print("All data loaded successfully!")
    
    return data


if __name__ == "__main__":
    # Quick test: load all data and print summary
    data = load_all_data()
    
    print("\n--- Sample Disease from Dataset ---")
    print(data['dataset'].iloc[0])
    
    print("\n--- Sample Description ---")
    print(data['descriptions'].iloc[0])
    
    print("\n--- Sample Precautions ---")
    print(data['precautions'].iloc[0])
    
    print("\n--- Sample Severity ---")
    print(data['severity'].head())
