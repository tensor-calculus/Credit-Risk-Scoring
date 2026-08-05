"""
06_explainability.py - SHAP-based Model Explainability
Generates global feature importance (beeswarm + bar) and per-applicant
waterfall plots using SHAP TreeExplainer on the LightGBM model.
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Sample size for SHAP computation (for performance)
SHAP_SAMPLE_SIZE = 5000


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load artifacts
    logging.info("Loading artifacts ...")
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    X_test = pd.read_parquet(ARTIFACTS_DIR / "X_test.parquet")
    y_test = pd.read_parquet(ARTIFACTS_DIR / "y_test.parquet")["TARGET"].values
    feature_names = joblib.load(ARTIFACTS_DIR / "feature_names.joblib")

    # Sample for SHAP if dataset is large
    if len(X_test) > SHAP_SAMPLE_SIZE:
        logging.info(f"Sampling {SHAP_SAMPLE_SIZE} from {len(X_test)} test samples for SHAP computation ...")
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_test), SHAP_SAMPLE_SIZE, replace=False)
        X_shap = X_test.iloc[sample_idx].copy()
        y_shap = y_test[sample_idx]
    else:
        X_shap = X_test.copy()
        y_shap = y_test
        sample_idx = np.arange(len(X_test))

    X_shap.columns = feature_names

    # SHAP TreeExplainer
    logging.info("Computing SHAP Values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap.values)

    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    print(f"  SHAP values shape: {shap_values_pos.shape}")

    # 1. Global beeswarm summary plot
    print("\nGenerating global summary plots...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_pos, X_shap.values,
        feature_names=feature_names,
        max_display=20, show=False, plot_type="dot"
    )
    plt.title("SHAP Feature Importance (Beeswarm)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots/shap_summary_beeswarm.png")

    # 2. Global bar summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_pos, X_shap.values,
        feature_names=feature_names,
        max_display=20, show=False, plot_type="bar"
    )
    plt.title("Mean |SHAP| Feature Importance", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots/shap_summary_bar.png")

    # 3. Per-applicant waterfall - defaulter
    print("\nGenerating per-applicant waterfall plots...")
    default_indices = np.where(y_shap == 1)[0]
    non_default_indices = np.where(y_shap == 0)[0]

    if len(default_indices) > 0:
        idx = default_indices[0]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]  # Class 1

        explanation = shap.Explanation(
            values=shap_values_pos[idx],
            base_values=expected_value,
            data=X_shap.values[idx],
            feature_names=feature_names
        )
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall — Defaulter", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_waterfall_default.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plots/shap_waterfall_default.png")

    if len(non_default_indices) > 0:
        idx = non_default_indices[0]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        explanation = shap.Explanation(
            values=shap_values_pos[idx],
            base_values=expected_value,
            data=X_shap.values[idx],
            feature_names=feature_names
        )
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall — Non-Defaulter", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_waterfall_nondefault.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plots/shap_waterfall_nondefault.png")

    # Save SHAP artifacts
    logging.info("Saving Artifacts ...")
    joblib.dump(shap_values_pos, ARTIFACTS_DIR / "shap_values.joblib")
    logging.info(f"Dumping shap_values.joblib (shape: {shap_values_pos.shape})")

    # Save sample indices so we can map SHAP values back to applicants
    joblib.dump(sample_idx, ARTIFACTS_DIR / "shap_sample_indices.joblib")
    logging.info(f"Dumping shap_sample_indices.joblib")

    logging.info("Done.")


if __name__ == "__main__":
    main()
