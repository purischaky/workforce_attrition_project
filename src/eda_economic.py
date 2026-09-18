"""
src/eda_economic.py
Econometric Exploratory Data Analysis & Hypothesis Testing
Curriculum Alignment: Hypothesis testing, behavioral economics drivers, risk metrics.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def run_econometric_eda(train_path: str = "data/processed/train.csv", reports_dir: str = "reports"):
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load strictly the training data to adhere to data governance
    df = pd.read_csv(train_path)
    
    print("=" * 60)
    print("ECONOMETRIC HYPOTHESIS TESTING (TRAIN SPLIT)")
    print("=" * 60)

    # Hypothesis 1: The Stagnation Penalty
    # H0: Mean promotion wait time is identical for staying and leaving employees.
    # H1: Leaving employees experience significantly longer wait times.
    stay_wait = df[df["attrition"] == 0]["promotion_wait_years"]
    leave_wait = df[df["attrition"] == 1]["promotion_wait_years"]
    t_stat_h1, p_val_h1 = stats.ttest_ind(leave_wait, stay_wait, equal_var=False)

    print("\n[Hypothesis 1: Career Stagnation Penalty]")
    print(f"  • Retained Mean Wait:   {stay_wait.mean():.2f} years")
    print(f"  • Departed Mean Wait:   {leave_wait.mean():.2f} years")
    print(f"  • Welch's t-statistic:  {t_stat_h1:.4f} | p-value: {p_val_h1:.4e}")
    if p_val_h1 < 0.01:
        print("  ✓ Result: Statistically significant at 1% level. Stagnation is a major turnover driver.")

    # Hypothesis 2: Effort-Reward Imbalance
    # H0: Workload per compensation ratio does not differ by attrition.
    df["effort_reward_ratio"] = (df["workload_score"] * (1 + df["overtime_hours_month"] / 40)) / (df["annual_salary_usd"] / 1000)
    stay_eff = df[df["attrition"] == 0]["effort_reward_ratio"]
    leave_eff = df[df["attrition"] == 1]["effort_reward_ratio"]
    t_stat_h2, p_val_h2 = stats.ttest_ind(leave_eff, stay_eff, equal_var=False)

    print("\n[Hypothesis 2: Effort-Reward Imbalance]")
    print(f"  • Retained Effort/Reward:  {stay_eff.mean():.4f}")
    print(f"  • Departed Effort/Reward:  {leave_eff.mean():.4f}")
    print(f"  • Welch's t-statistic:     {t_stat_h2:.4f} | p-value: {p_val_h2:.4e}")
    if p_val_h2 < 0.01:
        print("  ✓ Result: Statistically significant. Disproportionate effort vs compensation increases turnover hazard.")

    # Hypothesis 3: Manager Support Buffer
    # H0: Manager support score does not differ across groups.
    stay_mgr = df[df["attrition"] == 0]["manager_support_score"]
    leave_mgr = df[df["attrition"] == 1]["manager_support_score"]
    t_stat_h3, p_val_h3 = stats.ttest_ind(leave_mgr, stay_mgr, equal_var=False)

    print("\n[Hypothesis 3: Managerial Capital as a Retention Buffer]")
    print(f"  • Retained Manager Support: {stay_mgr.mean():.2f} / 10")
    print(f"  • Departed Manager Support: {leave_mgr.mean():.2f} / 10")
    print(f"  • Welch's t-statistic:      {t_stat_h3:.4f} | p-value: {p_val_h3:.4e}")
    if p_val_h3 < 0.01:
        print("  ✓ Result: Statistically significant. Managerial quality serves as a protective buffer.")

    # Visualization Generation
    generate_eda_plots(df, reports_dir)

def generate_eda_plots(df: pd.DataFrame, reports_dir: str):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Promotion Wait by Attrition
    sns.boxplot(
        data=df,
        x="attrition",
        y="promotion_wait_years",
        hue="attrition",
        legend=False,
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[0]
    )
    axes[0].set_title("H1: Career Stagnation vs. Attrition", fontsize=12, fontweight="bold")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Retained (0)", "Departed (1)"])
    axes[0].set_ylabel("Promotion Wait (Years)")
   

    # Plot 2: Workload vs Overtime by Attrition
    sns.scatterplot(
        data=df,
        x="workload_score",
        y="overtime_hours_month",
        hue="attrition",
        palette=["#2b5c8f", "#d95f02"],
        alpha=0.6,
        ax=axes[1]
    )
    axes[1].set_title("H2: Burnout Interaction (Workload & OT)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Perceived Workload (1-10)")
    axes[1].set_ylabel("Monthly Overtime (Hours)")

    # Plot 3: Manager Support Distribution
    sns.kdeplot(
        data=df,
        x="manager_support_score",
        hue="attrition",
        common_norm=False,
        fill=True,
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[2]
    )
    axes[2].set_title("H3: Manager Support Density", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Manager Support Score (1-10)")
    axes[2].set_ylabel("Density")

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "eda_economic_hypotheses.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n✓ High-resolution hypothesis visualization saved to: {plot_path}")

if __name__ == "__main__":
    run_econometric_eda()