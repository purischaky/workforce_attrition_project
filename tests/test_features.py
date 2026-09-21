"""
tests/test_features.py
Unit testing custom feature transformer logic using pytest.
Curriculum Alignment: Pytest fixtures, unit testing, edge case handling.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from src.features import LaborEconomicsFeatures

@pytest.fixture
def sample_employee_df():
    """Provides a synthetic batch of employee records including potential boundary edge cases."""
    return pd.DataFrame({
        "employee_id": ["EMP-001", "EMP-002", "EMP-003"],
        "department": ["Engineering", "Finance", "Engineering"],
        "job_level": ["Senior", "Junior", "Senior"],
        "workload_score": [8.0, 5.0, 9.0],
        "overtime_hours_month": [10.0, 0.0, 20.0],
        "manager_support_score": [0.0, 5.0, 10.0],  # EMP-001 tests zero-support division
        "tenure_years": [0.0, 2.0, 5.0],           # EMP-001 tests zero-tenure division
        "promotion_wait_years": [0.0, 1.0, 4.0],
        "promotion_count": [0, 1, 2],
        "annual_salary_usd": [90000, 50000, 95000],
    })

def test_zero_division_guard(sample_employee_df):
    """Asserts that zero-tenure and zero-support produce valid, finite floats."""
    transformer = LaborEconomicsFeatures()
    transformer.fit(sample_employee_df)
    transformed_df = transformer.transform(sample_employee_df)

    # Check finite outputs (no infs or NaNs)
    assert not np.isinf(transformed_df["burnout_pressure_ratio"]).any()
    assert not np.isnan(transformed_df["burnout_pressure_ratio"]).any()
    assert not np.isinf(transformed_df["career_velocity"]).any()
    assert not np.isnan(transformed_df["career_velocity"]).any()
    assert not np.isinf(transformed_df["stagnation_index"]).any()
    assert not np.isnan(transformed_df["stagnation_index"]).any()

def test_relative_salary_ratio(sample_employee_df):
    """Verifies that relative salary is calculated correctly based on median benchmarks."""
    transformer = LaborEconomicsFeatures()
    transformer.fit(sample_employee_df)
    transformed_df = transformer.transform(sample_employee_df)

    assert "relative_salary_ratio" in transformed_df.columns
    emp1_ratio = transformed_df.loc[0, "relative_salary_ratio"]
    assert pytest.approx(emp1_ratio, rel=1e-2) == 0.9729