"""
tests/test_pipeline.py
Unit tests verifying data leakage prevention and shape invariants.
Curriculum Alignment: Integration testing with pytest, data leakage audits.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from src.features import build_preprocessor_pipeline

def test_pipeline_leakage_isolation():
    """
    Verifies that pipeline transforms unseen test data strictly using training-derived statistics.
    """
    # Create distinct train and test dataframes
    train_data = pd.DataFrame({
        "department": ["Engineering"] * 10,
        "job_level": ["Senior"] * 10,
        "employment_type": ["Full-time"] * 10,
        "work_mode": ["Hybrid"] * 10,
        "age": [30] * 10,
        "tenure_years": [3.0] * 10,
        "annual_salary_usd": [100000] * 10,
        "workload_score": [5.0] * 10,
        "overtime_hours_month": [5.0] * 10,
        "manager_support_score": [5.0] * 10,
        "promotion_wait_years": [1.0] * 10,
        "promotion_count": [1] * 10,
        "training_hours_year": [20.0] * 10,
        "satisfaction_score": [7.0] * 10,
        "engagement_score": [7.0] * 10,
        "performance_score": [7.0] * 10,
        "absenteeism_days_year": [2.0] * 10,
    })

    test_data = train_data.copy()
    test_data["annual_salary_usd"] = [200000] * 10  # Wildly different distribution

    pipeline = build_preprocessor_pipeline()
    pipeline.fit(train_data)

    # Transform test set using the fitted pipeline
    X_test_trans = pipeline.transform(test_data)

    # Standard scaled column for salary should have a significant non-zero z-score
    salary_col_idx = 2  # 'annual_salary_usd' is index 2 in numeric features
    assert not np.allclose(X_test_trans[:, salary_col_idx], 0.0)

def test_pipeline_output_shape():
    """Ensures input dataframe columns map to consistent transformed dimensions."""
    train_df = pd.read_csv("data/processed/train.csv")
    pipeline = build_preprocessor_pipeline()
    transformed = pipeline.fit_transform(train_df)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(train_df)
    assert not np.isnan(transformed).any()