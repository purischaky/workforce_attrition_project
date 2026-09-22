"""
tests/test_models.py
Basic sanity checks for model training and calibration.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.models import train_and_evaluate

def test_train_and_evaluate_returns_valid_probabilities():
    """Ensures the pipeline trains without error and outputs valid probabilities."""
    pipeline, calibrated_hgb, hgb_raw, X_train_proc, X_test_proc, feature_names = train_and_evaluate()

    probs = calibrated_hgb.predict_proba(X_test_proc)[:, 1]

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert not np.isnan(probs).any()
    assert len(feature_names) == X_train_proc.shape[1]