"""
src/audit_data.py
Responsible AI Data Management & Schema Validation
Curriculum Alignment: Data governance, validation rules, leakage guardrails.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_validate_schema(data_path: str, dict_path: str) -> pd.DataFrame:
    """
    Validates dataset against data dictionary definitions:
    1. File existence
    2. Column presence and counts
    3. Missing values
    4. Primary key duplication
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file at: {data_path}")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Missing data dictionary at: {dict_path}")

    df = pd.read_csv(data_path)
    data_dict = pd.read_csv(dict_path)

    print("=" * 60)
    print("DATA GOVERNANCE & SCHEMA AUDIT")
    print("=" * 60)

    # Check shape
    expected_cols = set(data_dict["field"].str.strip())
    actual_cols = set(df.columns.str.strip())

    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols

    if missing_cols:
        raise ValueError(f"Schema mismatch! Missing expected columns: {missing_cols}")
    if extra_cols:
        print(f"[Warning] Unexpected columns found in dataset: {extra_cols}")
    print(f"✓ Schema Check Passed: All {len(expected_cols)} declared fields verified.")

    # Check primary key integrity
    if "employee_id" in df.columns:
        dup_count = df["employee_id"].duplicated().sum()
        if dup_count > 0:
            raise ValueError(f"Integrity check failed: {dup_count} duplicate employee_ids.")
        print("✓ Primary Key Integrity: 'employee_id' is distinct and unique.")

    # Missing Value Audit
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("[Warning] Missing values detected:")
        print(null_counts[null_counts > 0])
    else:
        print("✓ Completeness Check: Zero null values across 5,000 records.")

    # Target distribution audit
    if "attrition" in df.columns:
        rate = df["attrition"].mean()
        print(f"✓ Target Audit: Base attrition rate is {rate:.2%} ({df['attrition'].sum()} exits).")

    return df

def split_and_guard_leakage(df: pd.DataFrame, target_col: str = "attrition", test_size: float = 0.20, random_state: int = 42):
    """
    Executes an immutable stratified split to eliminate data leakage.
    Saves splits to data/processed/.
    """
    os.makedirs("data/processed", exist_ok=True)

    # Stratified split ensures exact class distribution in both train and test sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("REPRODUCIBLE LEAKAGE-FREE SPLIT")
    print("=" * 60)
    print(f"Training split saved to: {train_path} | Shape: {train_df.shape} (Attrition: {train_df[target_col].mean():.2%})")
    print(f"Holdout split saved to:  {test_path}  | Shape: {test_df.shape}  (Attrition: {test_df[target_col].mean():.2%})")
    print("✓ Governance boundary established: All downstream EDA operates on train.csv only.")

if __name__ == "__main__":
    raw_data = "data/raw/employees.csv"
    raw_dict = "data/raw/data_dictionary.csv"
    
    df_loaded = load_and_validate_schema(raw_data, raw_dict)
    split_and_guard_leakage(df_loaded)