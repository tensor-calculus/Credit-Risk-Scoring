"""
app.py - Credit Risk Scoring Dashboard
Two-tab dashboard:
  Tab 1: Applicant Lookup - score, risk band, SHAP waterfall
  Tab 2: Portfolio View  - score distribution, PSI, risk breakdown, model metrics
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Page config
st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Data loading (cached)
@st.cache_data
def load_scores():
    train_scores = pd.read_parquet(ARTIFACTS_DIR / "train_scores.parquet")
    test_scores = pd.read_parquet(ARTIFACTS_DIR / "test_scores.parquet")
    return train_scores, test_scores

@st.cache_data
def load_test_data():
    X_test = pd.read_parquet(ARTIFACTS_DIR / "X_test.parquet")
    y_test = pd.read_parquet(ARTIFACTS_DIR / "y_test.parquet")
    test_ids = pd.read_parquet(ARTIFACTS_DIR / "test_ids.parquet")
    return X_test, y_test, test_ids

@st.cache_data
def load_features():
    return pd.read_parquet(ARTIFACTS_DIR / "features.parquet")

@st.cache_data
def load_shap_values():
    shap_values = joblib.load(ARTIFACTS_DIR / "shap_values.joblib")
    shap_indices = joblib.load(ARTIFACTS_DIR / "shap_sample_indices.joblib")
    return shap_values, shap_indices

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS_DIR / "model.joblib")

@st.cache_resource
def load_expected_value():
    model = load_model()
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(model)
    return float(np.atleast_1d(explainer.expected_value)[-1])

@st.cache_data
def load_metrics():
    with open(ARTIFACTS_DIR / "model_metrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_feature_names():
    return joblib.load(ARTIFACTS_DIR / "feature_names.joblib")


def compute_psi(expected, actual, n_bins=10):
    """Compute Population Stability Index between two score distributions."""
    min_score = min(expected.min(), actual.min())
    max_score = max(expected.max(), actual.max())
    bins = np.linspace(min_score, max_score, n_bins + 1)

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = np.sum(psi_values)

    breakdown = pd.DataFrame({
        "Bin": [f"{bins[i]:.0f}–{bins[i+1]:.0f}" for i in range(n_bins)],
        "Train %": (expected_pct * 100).round(2),
        "Test %": (actual_pct * 100).round(2),
        "PSI": np.round(psi_values, 4),
    })

    return psi_total, breakdown


# Sidebar
st.sidebar.title("Credit Risk Scoring")
st.sidebar.divider()
st.sidebar.write(
    "End-to-end credit risk scoring pipeline with **LightGBM**, "
    "**PDO Scorecard**, and **SHAP Explainability**."
)

try:
    metrics = load_metrics()
    lgbm_metrics = metrics.get("lightgbm", {})
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric("AUC-ROC", f"{lgbm_metrics.get('auc_roc', 0):.4f}")
    st.sidebar.metric("Gini", f"{lgbm_metrics.get('gini', 0):.4f}")
    st.sidebar.metric("KS-Stat", f"{lgbm_metrics.get('ks_statistic', 0):.4f}")
except Exception:
    st.sidebar.warning("Metrics not loaded yet.")


# Main content
st.title("Credit Risk Scoring Dashboard")
tab1, tab2 = st.tabs(["Applicant Lookup", "Portfolio View"])


# ===== TAB 1: Applicant Lookup =====
with tab1:
    try:
        train_scores, test_scores = load_scores()
        X_test, y_test, test_ids = load_test_data()
        feature_names = load_feature_names()
        features_df = load_features()

        # Build dropdown options: SHAP sample IDs first, then remaining IDs
        try:
            shap_vals, shap_idx = load_shap_values()
            shap_ids = test_scores.iloc[shap_idx]["SK_ID_CURR"].tolist()
            other_ids = [i for i in test_scores["SK_ID_CURR"].tolist() if i not in set(shap_ids)]
            all_applicant_ids = shap_ids + other_ids
        except Exception:
            all_applicant_ids = test_scores["SK_ID_CURR"].tolist()

        selected_applicant_id = st.selectbox(
            "Select Applicant ID (SK_ID_CURR)",
            options=all_applicant_ids,
            index=0,
            help="Select an applicant to view their credit score, risk band, and SHAP feature contributions. Type to search."
        )

        if selected_applicant_id:
            applicant_id = int(selected_applicant_id)
            match = test_scores[test_scores["SK_ID_CURR"] == applicant_id]

            if len(match) == 0:
                st.error(f"Applicant {applicant_id} not found in test set.")
            else:
                row = match.iloc[0]
                score = int(row["score"])
                risk_band = row["risk_band"]
                probability = row["probability"]

                # --- Score, risk band, actual outcome ---
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Credit Score", f"{score}", help="Range: 300–900")
                with col2:
                    st.metric("Risk Band", risk_band)
                    st.caption(f"Default probability: {probability:.2%}")
                with col3:
                    test_idx = test_ids[test_ids["SK_ID_CURR"] == applicant_id].index
                    if len(test_idx) > 0:
                        actual = y_test.iloc[test_idx[0]]["TARGET"]
                        outcome = "Defaulted ❌" if actual == 1 else "Repaid ✅"
                    else:
                        outcome = "Unknown"
                    st.metric("Actual Outcome", outcome)

                # --- Applicant info ---
                st.subheader("Applicant Information")
                app_data = features_df[features_df["SK_ID_CURR"] == applicant_id]
                if len(app_data) > 0:
                    app_row = app_data.iloc[0]
                    info_cols = st.columns(4)
                    info_fields = [
                        ("Income", "AMT_INCOME_TOTAL", "₹{:,.0f}"),
                        ("Credit Amount", "AMT_CREDIT", "₹{:,.0f}"),
                        ("Annuity", "AMT_ANNUITY", "₹{:,.0f}"),
                        ("Age (years)", "DAYS_BIRTH", None),
                    ]
                    for i, (label, col, fmt) in enumerate(info_fields):
                        with info_cols[i]:
                            val = app_row.get(col, None)
                            if col == "DAYS_BIRTH" and val is not None:
                                val = f"{abs(val) / 365.25:.1f}"
                            elif fmt and val is not None:
                                val = fmt.format(val)
                            else:
                                val = str(val) if val is not None else "N/A"
                            st.metric(label, val)

                # --- SHAP waterfall ---
                st.subheader("SHAP Explanation")
                try:
                    shap_values, shap_indices = load_shap_values()
                    test_row_idx = match.index[0]
                    original_test_idx = (
                        test_scores.index.get_loc(test_row_idx)
                        if test_row_idx in test_scores.index
                        else None
                    )
                    shap_pos = (
                        np.where(shap_indices == original_test_idx)[0]
                        if original_test_idx is not None
                        else np.array([])
                    )

                    if len(shap_pos) > 0:
                        import shap as shap_lib

                        expected_value = load_expected_value()

                        sv = shap_values[shap_pos[0]]
                        explanation = shap_lib.Explanation(
                            values=sv,
                            base_values=expected_value,
                            data=X_test.iloc[original_test_idx].values
                            if original_test_idx is not None
                            else None,
                            feature_names=feature_names,
                        )
                        plt.close("all")
                        fig = plt.figure(figsize=(12, 6))
                        shap_lib.plots.waterfall(explanation, max_display=12, show=False)
                        fig = plt.gcf()
                        st.pyplot(fig)
                        plt.close("all")
                    else:
                        st.info(
                            "This applicant is not in the SHAP sample. "
                            "SHAP values were computed on a random subset of 5,000 test applicants."
                        )
                except Exception as e:
                    st.warning(f"SHAP visualization unavailable: {e}")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you've run all pipeline steps (01–06) before launching the dashboard.")


# ===== TAB 2: Portfolio View =====
with tab2:
    try:
        train_scores, test_scores = load_scores()

        # --- Score distribution ---
        st.subheader("Score Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=train_scores["score"], name="Train", opacity=0.6,
            marker_color="#4A90D9", nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=test_scores["score"], name="Test", opacity=0.6,
            marker_color="#E74C3C", nbinsx=50,
        ))
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Credit Score",
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Risk band + PSI ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Band Breakdown")
            risk_counts = test_scores["risk_band"].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={
                    "Very Low Risk": "#27ae60",
                    "Low Risk": "#2ecc71",
                    "Medium Risk": "#f39c12",
                    "High Risk": "#e74c3c",
                    "Very High Risk": "#c0392b",
                },
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Population Stability Index (PSI)")
            psi_total, psi_breakdown = compute_psi(
                train_scores["score"].values, test_scores["score"].values
            )

            if psi_total < 0.1:
                psi_text = "✅ Stable"
            elif psi_total < 0.25:
                psi_text = "⚠️ Moderate Shift"
            else:
                psi_text = "🚨 Significant Shift"

            st.metric("PSI", f"{psi_total:.4f}", help="< 0.1 stable, 0.1–0.25 moderate, > 0.25 significant")
            st.write(f"**Status**: {psi_text}")
            st.dataframe(psi_breakdown, use_container_width=True, hide_index=True)

        # --- Model metrics ---
        st.subheader("Model Performance Metrics")
        try:
            metrics = load_metrics()
            lgbm = metrics.get("lightgbm", {})

            m_cols = st.columns(5)
            metric_items = [
                ("AUC-ROC", lgbm.get("auc_roc", 0)),
                ("PR-AUC", lgbm.get("pr_auc", 0)),
                ("KS-Statistic", lgbm.get("ks_statistic", 0)),
                ("Gini", lgbm.get("gini", 0)),
                ("Best CV AUC", lgbm.get("best_cv_auc", 0)),
            ]
            for i, (label, value) in enumerate(metric_items):
                with m_cols[i]:
                    st.metric(label, f"{value:.4f}")
        except Exception:
            st.warning("Model metrics not available.")

        # --- Feature importance ---
        st.subheader("Feature Importance (Top 15)")
        try:
            model = load_model()
            feature_names = load_feature_names()
            importances = model.feature_importances_
            top_n = 15
            indices = np.argsort(importances)[-top_n:]
            fig = px.bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation="h",
                labels={"x": "Importance", "y": "Feature"},
                color=importances[indices],
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=450, showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure you've run all pipeline steps before launching the dashboard.")
