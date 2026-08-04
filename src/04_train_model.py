"""
04_train_model.py — Model training with Optuna HPO
Trains a Logistic Regression baseline and a LightGBM model with Optuna
hyperparameter optimization (50 trials, stratified 5-fold CV).
Reports AUC-ROC, PR-AUC, KS-statistic, and Gini.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg") # Anti-Grain Geometry := Non-interactive non-GUI backend of matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = PROJECT_ROOT / "plots"

RANDOM_STATE = 42
N_TRIALS = 50
N_FOLDS = 5


def load_data():
    """Load preprocessed train/test data."""
    X_train = pd.read_parquet(ARTIFACTS_DIR / "X_train.parquet")
    X_test = pd.read_parquet(ARTIFACTS_DIR / "X_test.parquet")
    y_train = pd.read_parquet(ARTIFACTS_DIR / "y_train.parquet")["TARGET"].values
    y_test = pd.read_parquet(ARTIFACTS_DIR / "y_test.parquet")["TARGET"].values
    feature_names = joblib.load(ARTIFACTS_DIR / "feature_names.joblib")
    return X_train.values, X_test.values, y_train, y_test, feature_names


def compute_ks_statistic(y_true, y_proba):
    """Compute the Kolmogorov-Smirnov statistic: max|TPR - FPR|."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(np.abs(tpr - fpr))
    return ks


def train_baseline(X_train, y_train, X_test, y_test):
    """Train Logistic Regression baseline and return metrics."""
    print("\n" + "=" * 60)
    print("Baseline: Logistic Regression")
    print("=" * 60)

    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    ks_stat = compute_ks_statistic(y_test, y_proba)
    gini = 2 * auc_roc - 1

    print(f"  AUC-ROC:      {auc_roc:.4f}")
    print(f"  PR-AUC:       {pr_auc:.4f}")
    print(f"  KS-Statistic: {ks_stat:.4f}")
    print(f"  Gini:         {gini:.4f}")

    metrics = {"auc_roc": auc_roc, "pr_auc": pr_auc, "ks_statistic": ks_stat, "gini": gini}
    return lr, y_proba, metrics


def optuna_objective(trial, X_train, y_train, scale_pos_weight):
    """Optuna objective: stratified 5-fold CV with LightGBM."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_pred))

    return np.mean(auc_scores)


def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    """Run Optuna HPO and train final LightGBM model."""
    print("\n" + "=" * 60)
    print(f"LightGBM with Optuna HPO ({N_TRIALS} trials, {N_FOLDS}-fold CV)")
    print("=" * 60)

    # Class imbalance weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight: {scale_pos_weight:.2f} (ratio {n_neg:,} neg / {n_pos:,} pos)")

    # Optuna study
    print(f"\n  Running {N_TRIALS} Optuna trials...")
    study = optuna.create_study(direction="maximize", study_name="lgbm_credit_risk")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\n  Best trial:")
    print(f"    Value (mean CV AUC): {study.best_value:.4f}")
    print(f"    Params: {study.best_params}")

    # Train final model on full training set with best params
    print("\n  Training final model on full training set...")
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
    })

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    y_proba = final_model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    ks_stat = compute_ks_statistic(y_test, y_proba)
    gini = 2 * auc_roc - 1

    print(f"\n  Test Set Metrics:")
    print(f"  {'AUC-ROC':<20} {auc_roc:.4f}")
    print(f"  {'PR-AUC':<20} {pr_auc:.4f}")
    print(f"  {'KS-Statistic':<20} {ks_stat:.4f}")
    print(f"  {'Gini':<20} {gini:.4f}")

    metrics = {
        "auc_roc": auc_roc, "pr_auc": pr_auc,
        "ks_statistic": ks_stat, "gini": gini,
        "best_cv_auc": study.best_value,
        "best_params": best_params,
    }
    return final_model, y_proba, metrics, study


def generate_plots(y_test, lr_proba, lgbm_proba, lgbm_model, feature_names):
    """Generate ROC, PR, and feature importance plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in [("Logistic Regression", lr_proba), ("LightGBM", lgbm_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/roc_curve.png")

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in [("Logistic Regression", lr_proba), ("LightGBM", lgbm_proba)]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", linewidth=2)
    baseline_rate = np.mean(y_test)
    ax.axhline(y=baseline_rate, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_rate:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/precision_recall_curve.png")

    # 3. Feature Importance (top 20)
    importances = lgbm_model.feature_importances_
    indices = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], align="center", color="#4A90D9")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance (split count)", fontsize=12)
    ax.set_title("Top 20 Feature Importances — LightGBM", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/feature_importance.png")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Baseline
    lr_model, lr_proba, lr_metrics = train_baseline(X_train, y_train, X_test, y_test)

    # LightGBM + Optuna
    lgbm_model, lgbm_proba, lgbm_metrics, study = train_lightgbm(
        X_train, y_train, X_test, y_test, feature_names
    )

    # Comparison table
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Logistic Reg':>15} {'LightGBM':>15}")
    print(f"  {'-'*50}")
    for metric in ["auc_roc", "pr_auc", "ks_statistic", "gini"]:
        print(f"  {metric:<20} {lr_metrics[metric]:>15.4f} {lgbm_metrics[metric]:>15.4f}")

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    generate_plots(y_test, lr_proba, lgbm_proba, lgbm_model, feature_names)

    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    joblib.dump(lgbm_model, ARTIFACTS_DIR / "model.joblib")
    print(f"  model.joblib (LightGBM)")
    joblib.dump(lr_model, ARTIFACTS_DIR / "baseline_model.joblib")
    print(f"  baseline_model.joblib (Logistic Regression)")
    joblib.dump(study, ARTIFACTS_DIR / "optuna_study.joblib")
    print(f"  optuna_study.joblib")

    # Save metrics as JSON (convert numpy types for serialization)
    all_metrics = {
        "logistic_regression": {k: float(v) for k, v in lr_metrics.items()},
        "lightgbm": {k: float(v) if isinstance(v, (float, np.floating)) else v
                     for k, v in lgbm_metrics.items()
                     if k != "best_params"},  # Exclude non-serializable params
        "lightgbm_best_params": {k: float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v
                                 for k, v in lgbm_metrics.get("best_params", {}).items()},
    }
    with open(ARTIFACTS_DIR / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  model_metrics.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
