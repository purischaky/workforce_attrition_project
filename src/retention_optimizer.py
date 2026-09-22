"""
src/retention_optimizer.py
Constrained Budget Optimization for Employee Retention Interventions.
Curriculum Alignment: Business Value Maximization, Expected Cost-Benefit Decision Framework.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from src.models import load_trained_artifacts

def optimize_retention_budget(total_budget: float = 50000.0, intervention_cost: float = 2500.0, success_rate: float = 0.40):
    print("\n" + "=" * 60)
    print("CONSTRAINED RETENTION ROI OPTIMIZATION")
    print("=" * 60)

    # Load test data and calibrated predictions
    test_df = pd.read_csv("data/processed/test.csv")
    pipeline, calibrated_hgb, _, _ = load_trained_artifacts()
    X_test_proc = pipeline.transform(test_df)

    test_df["predicted_attrition_prob"] = calibrated_hgb.predict_proba(X_test_proc)[:, 1]

    # Calculate financial metrics (Replacement Cost = 1.5 * Annual Salary)
    test_df["replacement_cost"] = test_df["annual_salary_usd"] * 1.5
    test_df["expected_loss"] = test_df["predicted_attrition_prob"] * test_df["replacement_cost"]
    test_df["expected_saved_value"] = test_df["expected_loss"] * success_rate
    test_df["net_expected_gain"] = test_df["expected_saved_value"] - intervention_cost

    # Filter candidates with positive net ROI and sort by net value saved
    eligible_candidates = test_df[test_df["net_expected_gain"] > 0].sort_values(
        by="net_expected_gain", ascending=False
    ).copy()

    max_interventions = int(total_budget // intervention_cost)
    selected_targets = eligible_candidates.head(max_interventions)

    total_cost = len(selected_targets) * intervention_cost
    total_expected_saved = selected_targets["expected_saved_value"].sum()
    net_roi = ((total_expected_saved - total_cost) / total_cost) * 100 if total_cost > 0 else 0.0

    print(f"\n[Budget Allocation Summary]")
    print(f"  • Available Budget:          ${total_budget:,.2f}")
    print(f"  • Cost Per Intervention:     ${intervention_cost:,.2f}")
    print(f"  • Interventions Allocated:   {len(selected_targets)} / {max_interventions} max")
    print(f"  • Total Capital Deployed:    ${total_cost:,.2f}")
    print(f"  • Expected Saved Value:      ${total_expected_saved:,.2f}")
    print(f"  • Portfolio Expected Net ROI:{net_roi:.1f}%")

    print("\n[Top Target Employee Interventions]")
    display_cols = ["employee_id", "department", "annual_salary_usd", "predicted_attrition_prob", "net_expected_gain"]
    print(selected_targets[display_cols].head(5).to_string(index=False))

    return selected_targets

if __name__ == "__main__":
    optimize_retention_budget()