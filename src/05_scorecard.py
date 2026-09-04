"""
05_scorecard.py - PDO Scorecard Mapping
Maps predicted default probabilities to a 300-850 credit score using
Points to Double the Odds (PDO) methodology with scale_pos_weight calibration.
Parameters:
  Base score = 650, Base odds = 11.5:1 (good:bad), PDO = 40, Range = [300, 850]
  Score = Offset + Factor * ln(odds)
  where odds = (1 - p) / p, Factor = PDO / ln(2), Offset = base_score - Factor * ln(base_odds)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = PROJECT_ROOT / "plots"

# PDO scorecard parameters
BASE_SCORE = 650
BASE_ODDS = 11.5   # ~8% default rate corresponds to ~11.5:1 good to bad odds
PDO = 40          # Points to Double the Odds
MIN_SCORE = 300
MAX_SCORE = 850


def calibrate_probability(prob, scale_pos_weight):
    """
    Calibrate model pseudo-probabilities back to true empirical probabilities
    using Bayes odds correction for scale_pos_weight:
      p_true = p / (p + (1 - p) * scale_pos_weight)
    """
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return p / (p + (1.0 - p) * scale_pos_weight)


def probability_to_score(probability, base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO, min_score=MIN_SCORE, max_score=MAX_SCORE):
    """
    Convert calibrated default probability to credit score using PDO methodology.

    Higher score = lower risk (better).
    Score = Offset + Factor * ln(odds)
    where odds = P(good) / P(bad) = (1 - p) / p
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    p = np.clip(probability, 1e-7, 1 - 1e-7)
    odds = (1.0 - p) / p
    score = offset + factor * np.log(odds)
    return np.clip(np.round(score).astype(int), min_score, max_score)


def assign_risk_band(score):
    """Assign risk band based on standard credit score tiers (300-850 range)."""
    if isinstance(score, (np.ndarray, pd.Series)):
        conditions = [
            score < 580,
            (score >= 580) & (score < 640),
            (score >= 640) & (score < 700),
            (score >= 700) & (score < 750),
            score >= 750,
        ]
        choices = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk", "Very Low Risk"]
        return np.select(conditions, choices, default="Unknown")
    else:
        if score < 580: return "Very High Risk"
        if score < 640: return "High Risk"
        if score < 700: return "Medium Risk"
        if score < 750: return "Low Risk"
        return "Very Low Risk"


def verify_monotonicity(scores, y_true):
    """Verify score monotonicity: higher score deciles should have lower default rates."""
    df = pd.DataFrame({"score": scores, "target": y_true})
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")

    decile_stats = df.groupby("decile").agg(
        count=("target", "count"),
        default_count=("target", "sum"),
        default_rate=("target", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        avg_score=("score", "mean"),
    ).reset_index()

    print("\n  Decile Analysis (higher decile = higher score = lower risk):")
    print(f"  {'Decile':>7} {'Count':>8} {'Defaults':>10} {'Default %':>10} {'Score Range':>18}")
    print(f"  {'-'*55}")

    is_monotone = True
    prev_rate = None
    for _, row in decile_stats.iterrows():
        rate = row["default_rate"]
        if prev_rate is not None and rate > prev_rate + 0.005:  # Small tolerance
            is_monotone = False
        prev_rate = rate
        print(f"  {int(row['decile']):>7} {int(row['count']):>8} {int(row['default_count']):>10} "
              f"{rate:>9.2%} {int(row['min_score']):>6}–{int(row['max_score'])}")

    if is_monotone:
        logging.info("Monotonicity verified: default rate decreases as score increases")
    else:
        logging.warning("Slight monotonicity violation detected (may be acceptable in practice)")

    return decile_stats


def generate_plots(train_scores, test_scores, y_test, test_deciles):
    """Generate score distribution and decile plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Score distribution (train vs test)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(train_scores, bins=50, alpha=0.5, label="Train", color="#4A90D9", density=True)
    ax.hist(test_scores, bins=50, alpha=0.5, label="Test", color="#E74C3C", density=True)
    ax.set_xlabel("Credit Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution — Train vs Test", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/score_distribution.png")

    # 2. Decile-wise default rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(test_deciles["decile"], test_deciles["default_rate"] * 100,
           color="#E74C3C", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Score Decile (0=Lowest, 9=Highest)", fontsize=12)
    ax.set_ylabel("Default Rate (%)", fontsize=12)
    ax.set_title("Default Rate by Score Decile", fontsize=14)
    ax.set_xticks(test_deciles["decile"])
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "decile_default_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/decile_default_rate.png")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load artifacts
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    X_train = pd.read_parquet(ARTIFACTS_DIR / "X_train.parquet")
    X_test = pd.read_parquet(ARTIFACTS_DIR / "X_test.parquet")
    y_train = pd.read_parquet(ARTIFACTS_DIR / "y_train.parquet")["TARGET"].values
    y_test = pd.read_parquet(ARTIFACTS_DIR / "y_test.parquet")["TARGET"].values
    train_ids = pd.read_parquet(ARTIFACTS_DIR / "train_ids.parquet")["SK_ID_CURR"].values
    test_ids = pd.read_parquet(ARTIFACTS_DIR / "test_ids.parquet")["SK_ID_CURR"].values

    # Compute scale_pos_weight from training set
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos

    # Predict raw model probabilities
    train_proba_raw = model.predict_proba(X_train.values)[:, 1]
    test_proba_raw = model.predict_proba(X_test.values)[:, 1]

    # Calibrate probabilities to reflect true empirical default rates
    train_proba = calibrate_probability(train_proba_raw, scale_pos_weight)
    test_proba = calibrate_probability(test_proba_raw, scale_pos_weight)

    # Convert to scores
    print("=" * 60)
    print("PDO Scorecard Mapping (Calibrated)")
    print("=" * 60)
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    print(f"  Scale Pos Weight: {scale_pos_weight:.2f}")
    print(f"  Base Score:       {BASE_SCORE}")
    print(f"  Base Odds:        {BASE_ODDS}:1")
    print(f"  PDO:              {PDO}")
    print(f"  Factor:           {factor:.4f}")
    print(f"  Offset:           {offset:.4f}")
    print(f"  Score Range:      [{MIN_SCORE}, {MAX_SCORE}]")

    train_scores = probability_to_score(train_proba)
    test_scores = probability_to_score(test_proba)

    # Score distribution statistics
    print(f"\n  Score Distribution:")
    for name, scores in [("Train", train_scores), ("Test", test_scores)]:
        print(f"    {name}: min={scores.min()}, max={scores.max()}, "
              f"mean={scores.mean():.0f}, std={scores.std():.0f}, "
              f"p25={np.percentile(scores, 25):.0f}, p50={np.percentile(scores, 50):.0f}, "
              f"p75={np.percentile(scores, 75):.0f}")

    # Risk band distribution
    train_bands = assign_risk_band(train_scores)
    test_bands = assign_risk_band(test_scores)

    print(f"\n  Risk Band Distribution (Test Set):")
    for band in ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]:
        count = np.sum(test_bands == band)
        pct = count / len(test_bands) * 100
        print(f"    {band:<20} {count:>8,} ({pct:>5.1f}%)")

    # Monotonicity check
    logging.info("Monotonicity Verification ...")
    decile_stats = verify_monotonicity(test_scores, y_test)

    # Generate plots
    logging.info("Generating Plots ...")
    generate_plots(train_scores, test_scores, y_test, decile_stats)

    # Save artifacts
    logging.info("Saving Artifacts ...")

    # Scorecard config
    scorecard_config = {
        "base_score": BASE_SCORE,
        "base_odds": BASE_ODDS,
        "pdo": PDO,
        "factor": factor,
        "offset": offset,
        "scale_pos_weight": scale_pos_weight,
        "min_score": MIN_SCORE,
        "max_score": MAX_SCORE,
    }
    joblib.dump(scorecard_config, ARTIFACTS_DIR / "scorecard.joblib")
    logging.info(f"Dumping scorecard.joblib ...")

    # Scores DataFrames
    train_scores_df = pd.DataFrame({
        "SK_ID_CURR": train_ids,
        "probability": train_proba,
        "score": train_scores,
        "risk_band": train_bands,
    })
    test_scores_df = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "probability": test_proba,
        "score": test_scores,
        "risk_band": test_bands,
    })
    train_scores_df.to_parquet(ARTIFACTS_DIR / "train_scores.parquet", index=False)
    test_scores_df.to_parquet(ARTIFACTS_DIR / "test_scores.parquet", index=False)
    logging.info(f"train_scores.parquet ({len(train_scores_df):,} rows)")
    logging.info(f"test_scores.parquet ({len(test_scores_df):,} rows)")

    logging.info("Done.")


if __name__ == "__main__":
    main()
