"""
main.py
End-to-End Project Execution Pipeline.
Orchestrates raw data validation, feature preprocessing, model calibration, SHAP explainability, segmentation, and retention optimization.
"""

import sys
import os

from src.audit_data import run_data_audit
from src.models import train_and_evaluate
from src.explainability import generate_shap_insights
from src.segmentation import run_segmentation
from src.retention_optimizer import optimize_retention_budget

def run_full_pipeline():
    print("=" * 80)
    print("STARTING WORKFORCE ATTRITION END-TO-END INTELLIGENCE PIPELINE")
    print("=" * 80)

    # Data Audit & Validation
    print("\n[STEP 1/5] Running Responsible AI Data Audit...")
    run_data_audit("data/raw/employees.csv")

    # Dual Model Calibration & Evaluation
    print("\n[STEP 2/5] Fitting Models & Calibrating Probabilities...")
    train_and_evaluate()

    # Explainable AI & SHAP Attributions
    print("\n[STEP 3/5] Computing Global & Local SHAP Values...")
    generate_shap_insights()

    # Latent Clustering
    print("\n[STEP 4/5] Executing PCA & K-Means Clustering...")
    run_segmentation()

    # Constrained Retention ROI Optimization
    print("\n[STEP 5/5] Optimizing Retention Budget Allocation...")
    optimize_retention_budget(total_budget=50000.0, intervention_cost=2500.0, success_rate=0.40)

    print("\n" + "=" * 80)
    print("✓ FULL PIPELINE EXECUTION COMPLETE - ALL ARTIFACTS SAVED TO reports/")
    print("=" * 80)

if __name__ == "__main__":
    run_full_pipeline()