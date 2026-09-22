"""
src/explainability.py
Model-Agnostic and Tree-Based Explainable AI via SHAP
Curriculum Alignment: Feature Attribution, Global Summary Beeswarm, Local Explainability.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from src.models import load_trained_artifacts


def generate_shap_insights(reports_dir: str = "reports"):
    os.makedirs(reports_dir, exist_ok=True)

    # Load previously trained models instead of retraining
    pipeline, calibrated_model, raw_tree_model, feature_names = load_trained_artifacts()

    test_df = pd.read_csv("data/processed/test.csv")
    X_test_proc = pipeline.transform(test_df)
    
    print("\n" + "=" * 60)
    print("EXPLAINABLE AI & SHAP FEATURE ATTRIBUTION")
    print("=" * 60)

    # Convert processed arrays back to DataFrames with named columns for clean SHAP plots
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(raw_tree_model)
    shap_values = explainer(X_test_df)

    # Global Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("SHAP Global Summary: Feature Contributions to Attrition Risk", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    summary_path = os.path.join(reports_dir, "shap_global_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Global SHAP summary plot saved to: {summary_path}")

    # Mean Absolute SHAP Feature Importance Table
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP_Impact": mean_abs_shap
    }).sort_values(by="Mean_SHAP_Impact", ascending=False)

    print("\n[Top 5 Global Turnover Risk Drivers by SHAP Impact]")
    print(shap_importance.head(5).to_string(index=False))

if __name__ == "__main__":
    generate_shap_insights()