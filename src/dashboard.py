"""
src/dashboard.py
Interactive Executive Dashboard for Workforce Attrition Risk & Retention ROI.
Run with: streamlit run src/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.models import train_and_evaluate
from src.retention_optimizer import optimize_retention_budget
from src.segmentation import run_segmentation

st.set_page_config(
    page_title="Workforce Attrition Intelligence & Retention Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Workforce Attrition Intelligence & Retention Optimization")
st.markdown("Executive decision support platform for risk scoring, archetype segmentation, and budget allocation.")

# Sidebar Configuration
st.sidebar.header("🕹️ Optimization Settings")
budget_input = st.sidebar.number_input("Total Retention Budget ($)", min_value=5000, max_value=500000, value=50000, step=5000)
cost_input = st.sidebar.number_input("Intervention Cost per Employee ($)", min_value=500, max_value=20000, value=2500, step=500)
success_rate_input = st.sidebar.slider("Expected Retention Success Rate", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

# Tabs Navigation
tab1, tab2, tab3 = st.tabs(["💰 Budget ROI Optimizer", "🧩 Latent Workforce Clusters", "🔍 Risk Drivers & Explainability"])

with tab1:
    st.subheader("Constrained Budget Allocation")
    
    if st.button("Run Portfolio Optimization", type="primary"):
        targets = optimize_retention_budget(
            total_budget=float(budget_input),
            intervention_cost=float(cost_input),
            success_rate=float(success_rate_input)
        )
        
        st.success("Successfully allocated interventions for top candidates!")
        st.dataframe(targets, width="stretch")

with tab2:
    st.subheader("PCA + K-Means Latent Archetypes")
    if os.path.exists("reports/workforce_clusters_pca.png"):
        st.image("reports/workforce_clusters_pca.png", caption="Workforce Latent Clusters (PCA Projection)")
    else:
        st.info("Run `python src/segmentation.py` to generate the cluster plot.")

with tab3:
    st.subheader("Global SHAP Risk Driver Attribution")
    if os.path.exists("reports/shap_global_summary.png"):
        st.image("reports/shap_global_summary.png", caption="Global SHAP Beeswarm Plot")
    else:
        st.info("Run `python src/explainability.py` to generate the SHAP report.")