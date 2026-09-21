"""
src/models.py
Dual Modeling Framework: Econometric Baseline & Gradient Boosted Ensemble with Probability Calibration
Curriculum Alignment: Logistic Regression, Odds Ratios, ROC Curves, Hyperparameter Tuning, Calibration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from src.features import build_preprocessor_pipeline, get_feature_lists

def get_feature_names(pipeline) -> list:
    """Extracts explicit feature names post-ColumnTransformer encoding."""
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_cols, num_cols = get_feature_lists()
    
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
    
    return num_cols + encoded_cat_cols

def train_and_evaluate():
    print("=" * 60)
    print("DUAL MODELING & PROBABILITY CALIBRATION")
    print("=" * 60)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    y_train = train_df["attrition"]
    y_test = test_df["attrition"]

    # Fit Preprocessing Pipeline
    pipeline = build_preprocessor_pipeline()
    X_train_proc = pipeline.fit_transform(train_df)
    X_test_proc = pipeline.transform(test_df)
    feature_names = get_feature_names(pipeline)

    # Econometric Model: Logistic Regression (Odds Ratios)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_proc, y_train)
    lr_probs = lr.predict_proba(X_test_proc)[:, 1]

    # Compute Odds Ratios
    odds_ratios = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lr.coef_[0],
        "Odds_Ratio": np.exp(lr.coef_[0])
    }).sort_values(by="Odds_Ratio", ascending=False)

    print("\n[Econometric Baseline: Logistic Regression Top Risk Drivers]")
    print(odds_ratios.head(5).to_string(index=False))

    # Production Model: HistGradientBoosting
    hgb_raw = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    hgb_raw.fit(X_train_proc, y_train)

    # Calibrated Model (5-Fold Cross-Validation Calibration)
    calibrated_hgb = CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        method="sigmoid",
        cv=5
    )
    calibrated_hgb.fit(X_train_proc, y_train)
    hgb_probs = calibrated_hgb.predict_proba(X_test_proc)[:, 1]

    # Metrics Evaluation
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_brier = brier_score_loss(y_test, lr_probs)
    
    hgb_auc = roc_auc_score(y_test, hgb_probs)
    hgb_brier = brier_score_loss(y_test, hgb_probs)

    print("\n[Model Performance Comparison on Holdout Test Set]")
    print(f"  • Logistic Regression  | ROC-AUC: {lr_auc:.4f} | Brier Score: {lr_brier:.4f}")
    print(f"  • Calibrated LightGBM  | ROC-AUC: {hgb_auc:.4f} | Brier Score: {hgb_brier:.4f}")
    print("✓ Model training and probability calibration complete.")

    return pipeline, calibrated_hgb, hgb_raw, X_train_proc, X_test_proc, feature_names

if __name__ == "__main__":
    train_and_evaluate()