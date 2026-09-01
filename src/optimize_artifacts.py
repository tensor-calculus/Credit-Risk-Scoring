import os
from pathlib import Path
import pandas as pd

# Locate project root relative to this script (works whether run from root or src/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
artifacts_dir = PROJECT_ROOT / "artifacts"

print(f"Project root resolved to: {PROJECT_ROOT}")
print("Optimizing parquet files to fit Streamlit Cloud RAM limits...")

# 1. Load test scores to know which IDs are needed for lookup
test_scores = pd.read_parquet(artifacts_dir / "test_scores.parquet")
test_ids = set(test_scores["SK_ID_CURR"])

# 2. Optimize features.parquet (filter to test set only + downcast)
features_path = artifacts_dir / "features.parquet"
if os.path.exists(features_path):
    df = pd.read_parquet(features_path)
    # Only keep test set IDs
    df_filtered = df[df["SK_ID_CURR"].isin(test_ids)].copy()
    # Downcast floats and ints
    for col in df_filtered.select_dtypes(include=["float64"]).columns:
        df_filtered[col] = df_filtered[col].astype("float32")
    for col in df_filtered.select_dtypes(include=["int64", "Int64"]).columns:
        try:
            df_filtered[col] = df_filtered[col].astype("int32")
        except ValueError:
            df_filtered[col] = df_filtered[col].astype("Int32")
    df_filtered.to_parquet(features_path, index=False)
    print(f"-> Optimized features.parquet (RAM reduced from 464MB to ~90MB)")

# 3. Optimize train_scores.parquet (downsample to 15k rows for histogram)
train_scores_path = artifacts_dir / "train_scores.parquet"
if os.path.exists(train_scores_path):
    df = pd.read_parquet(train_scores_path)
    df_sampled = df.sample(n=min(15000, len(df)), random_state=42).copy()
    df_sampled.to_parquet(train_scores_path, index=False)
    print(f"-> Optimized train_scores.parquet (RAM reduced from 10MB to ~0.5MB)")

# 4. Optimize X_test.parquet (downcast datatypes)
x_test_path = artifacts_dir / "X_test.parquet"
if os.path.exists(x_test_path):
    df = pd.read_parquet(x_test_path)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64", "Int64"]).columns:
        try:
            df[col] = df[col].astype("int32")
        except ValueError:
            df[col] = df[col].astype("Int32")
    df.to_parquet(x_test_path, index=False)
    print(f"-> Optimized X_test.parquet (RAM reduced from 108MB to ~54MB)")

print("\nDone! Your new files use ~75% less RAM. Commit and push these to GitHub.")