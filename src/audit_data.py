"""
src/audit_data.py
Responsible AI & Data Quality Audit for Workforce Attrition Dataset.
Curriculum Alignment: Responsible AI (Fairness Audit, Data Profiling, Missingness Analysis).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np

def run_data_audit(filepath: str = None) -> pd.DataFrame:
    print("=" * 60)
    print("RESPONSIBLE AI & DATA QUALITY AUDIT")
    print("=" * 60)

    # Dynamic file fallback check
    candidate_paths = [
        filepath,
        "data/raw/workforce_data.csv",
        "data/processed/train.csv"
    ]
    
    valid_path = None
    for path in candidate_paths:
        if path and os.path.exists(path):
            valid_path = path
            break

    if not valid_path:
        raise FileNotFoundError("Could not locate workforce dataset in 'data/raw/' or 'data/processed/'.")

    print(f"✓ Auditing dataset file: {valid_path}")
    df = pd.read_csv(valid_path)

    print(f"✓ Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    # Missing values check
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("✓ Missing Values: None detected.")
    else:
        print("\n⚠ Missing Values Detected:")
        print(missing)

    # Duplicates check
    duplicates = df.duplicated().sum()
    print(f"✓ Duplicate Rows: {duplicates}")

    # Class balance check
    if "attrition" in df.columns:
        attrition_counts = df["attrition"].value_counts(normalize=True) * 100
        print("\n[Target Variable: Attrition Distribution]")
        for val, pct in attrition_counts.items():
            print(f"  • Class {val}: {pct:.2f}%")

    print("\n✓ Data Audit Complete.")
    return df

if __name__ == "__main__":
    run_data_audit()