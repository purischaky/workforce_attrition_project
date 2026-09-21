"""
src/features.py
Labor Economics Feature Engineering & Preprocessing Pipelines
Curriculum Alignment: Custom Transformers, Pipelines, Centering/Scaling, One-Hot Encoding.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class LaborEconomicsFeatures(BaseEstimator, TransformerMixin):
    """
    Transforms baseline HR metrics into structural labor-economic ratios:
    1. Burnout Pressure Ratio: (Workload * Overtime) / (Manager Support + epsilon)
    2. Career Velocity: (Promotions + 1) / (Tenure + 0.5)
    3. Stagnation Index: Wait Years / (Tenure + 0.1)
    4. Relative Salary Ratio: Individual Salary / Department-Level Median Salary
    """

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.reference_medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        # Learn reference median salaries strictly from the training partition to prevent leakage
        df = X.copy()
        if "department" in df.columns and "job_level" in df.columns and "annual_salary_usd" in df.columns:
            group_medians = df.groupby(["department", "job_level"])["annual_salary_usd"].median()
            self.reference_medians_ = group_medians.to_dict()
            self.overall_median_salary_ = df["annual_salary_usd"].median()
        else:
            self.overall_median_salary_ = 75000.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Burnout Pressure Ratio
        df["burnout_pressure_ratio"] = (
            df["workload_score"] * df["overtime_hours_month"]
        ) / (df["manager_support_score"] + self.epsilon)

        # 2. Career Velocity & Stagnation Index
        df["career_velocity"] = (df["promotion_count"] + 1.0) / (df["tenure_years"] + 0.5)
        df["stagnation_index"] = df["promotion_wait_years"] / (df["tenure_years"] + 0.1)

        # 3. Relative Salary Ratio (Benchmark against internal peers)
        def get_benchmark(row):
            key = (row.get("department"), row.get("job_level"))
            return self.reference_medians_.get(key, self.overall_median_salary_)

        benchmarks = df.apply(get_benchmark, axis=1)
        df["relative_salary_ratio"] = df["annual_salary_usd"] / (benchmarks + self.epsilon)

        return df

def get_feature_lists():
    """Defines feature types for the pipeline."""
    categorical_features = ["department", "job_level", "employment_type", "work_mode"]
    
    numeric_features = [
        "age",
        "tenure_years",
        "annual_salary_usd",
        "workload_score",
        "overtime_hours_month",
        "manager_support_score",
        "promotion_wait_years",
        "promotion_count",
        "training_hours_year",
        "satisfaction_score",
        "engagement_score",
        "performance_score",
        "absenteeism_days_year",
        "burnout_pressure_ratio",
        "career_velocity",
        "stagnation_index",
        "relative_salary_ratio",
    ]
    return categorical_features, numeric_features

def build_preprocessor_pipeline() -> Pipeline:
    """
    Assembles complete scikit-learn preprocessing pipeline:
    LaborEconomicsFeatures -> ColumnTransformer(StandardScaler + OneHotEncoder).
    """
    categorical_features, numeric_features = get_feature_lists()

    transformer_pipeline = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("labor_features", LaborEconomicsFeatures()),
            ("preprocessor", transformer_pipeline),
        ]
    )
    return full_pipeline

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    pipe = build_preprocessor_pipeline()
    X_proc = pipe.fit_transform(train_df)
    print("=" * 60)
    print("PIPELINE TEST RUN")
    print("=" * 60)
    print(f"✓ Pipeline fitted and transformed successfully.")
    print(f"✓ Output feature matrix shape: {X_proc.shape}")