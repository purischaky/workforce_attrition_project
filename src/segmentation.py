"""
src/segmentation.py
Unsupervised Workforce Segmentation using PCA and K-Means Clustering.
Curriculum Alignment: Unsupervised Learning (Dimensionality Reduction, Archetype Profiling).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.features import build_preprocessor_pipeline

def run_segmentation(n_clusters: int = 4, reports_dir: str = "reports"):
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 60)
    print("LATENT WORKFORCE SEGMENTATION (PCA + K-MEANS)")
    print("=" * 60)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Pipeline Transformation
    pipeline = build_preprocessor_pipeline()
    X_proc = pipeline.fit_transform(full_df)

    # PCA Dimensionality Reduction
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_proc)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"✓ PCA 2-Component Explained Variance: {explained_var:.2f}%")

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_proc)

    full_df["cluster"] = cluster_labels
    full_df["pca_1"] = X_pca[:, 0]
    full_df["pca_2"] = X_pca[:, 1]

    # Plot Cluster Segmentation Map
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        full_df["pca_1"],
        full_df["pca_2"],
        c=full_df["cluster"],
        cmap="Set2",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5
    )
    plt.colorbar(scatter, label="Cluster Archetype")
    plt.title("Workforce Latent Clusters (PCA Projection)", fontsize=12, fontweight="bold")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()

    fig_path = os.path.join(reports_dir, "workforce_clusters_pca.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Cluster visualization saved to: {fig_path}")

    # Cluster Summary Statistics
    cluster_summary = full_df.groupby("cluster").agg(
        Count=("employee_id", "count"),
        Attrition_Rate=("attrition", "mean"),
        Avg_Annual_Salary=("annual_salary_usd", "mean"),
        Avg_Workload=("workload_score", "mean"),
        Avg_Satisfaction=("satisfaction_score", "mean")
    ).reset_index()

    print("\n[Workforce Cluster Profiles]")
    print(cluster_summary.to_string(index=False))

    return full_df, cluster_summary

if __name__ == "__main__":
    run_segmentation()