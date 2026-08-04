"""
03_preprocessing.py - Build sklearn preprocessing pipeline
Performs leakage audit, train/test split, and constructs a ColumnTransformer
pipeline with proper encoding for numeric and categorical features.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import mlflow
import logging

# logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42

def load_data() -> pd.DataFrame:
    """Load the feature table from parquet."""
    path = ARTIFACTS_DIR / "features.parquet"
    logging.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    logging.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def leakage_audit(df: pd.DataFrame) -> None:
    """Check for target leakage and print findings to console."""
    print("\n" + "=" * 70)
    print(" Leakage Audit")
    print("=" * 70)

    y = df["TARGET"]

    # Check each numeric column for suspiciously high AUC
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("SK_ID_CURR", "TARGET")]

    suspicious = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) < 100:
            continue
        idx = vals.index
        try:
            auc = roc_auc_score(y.loc[idx], vals)
            auc = max(auc, 1 - auc)  # Handle reversed direction
            if auc > 0.99:
                suspicious.append((col, auc))
        except Exception:
            pass

    if suspicious:
        logging.warning("Potential leakage detected:")
        for col, auc in suspicious:
            print(f"  {col}: AUC = {auc:.4f}")
    else:
        logging.info("No single feature has AUC > 0.99 against TARGET")


def identify_column_types(X: pd.DataFrame):
    """Split columns into numeric, low-cardinality categorical, and high-cardinality categorical columns."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    low_card_cats = [c for c in cat_cols if X[c].nunique() < 10]
    high_card_cats = [c for c in cat_cols if X[c].nunique() >= 10]

    print("=" * 70)
    print(f"  Column Types: ")
    print("=" * 70)
    print(f"  Numeric features:             {len(numeric_cols)}")
    print(f"  Low-cardinality categorical:  {len(low_card_cats)} (< 10 unique values)")
    print(f"  High-cardinality categorical: {len(high_card_cats)} (>= 10 unique values)")

    if high_card_cats:
        print(f"High Cardinality Columns:")
        print("-" * 70)
        for c in high_card_cats:
            print(f"  {c}: {X[c].nunique()} unique values")

    return numeric_cols, low_card_cats, high_card_cats


def build_preprocessor(numeric_cols, low_card_cats, high_card_cats) -> ColumnTransformer:
    """Build the ColumnTransformer pipeline."""
    # Numeric: Impute median -> Scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Low-cardinality categorical: Impute constant -> OneHot
    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # High-cardinality categorical: TargetEncoder
    # TargetEncoder handles NaNs natively.
    high_card_pipeline = Pipeline([
        ("target_enc", TargetEncoder(smooth="auto", cv=5))
    ])

    transformers = [
        ("numeric", numeric_pipeline, numeric_cols),
    ]
    if low_card_cats:
        transformers.append(("low_card_cat", low_card_pipeline, low_card_cats))
    if high_card_cats:
        transformers.append(("high_card_cat", high_card_pipeline, high_card_cats))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def main() -> None:
    df = load_data()
    leakage_audit(df)

    # Separate features and target
    logging.info("Separating features & target ...")
    ids = df["SK_ID_CURR"]
    y = df["TARGET"]
    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    logging.info(f"Features: {X.shape[1]} columns")
    logging.info(f"Target: {y.shape[0]:,} samples")

    # Identify column types
    numeric_cols, low_card_cats, high_card_cats = identify_column_types(X)

    # Train/test split
    logging.info("Calculating Train/Test split (80/20, stratified) ...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"Train: {X_train.shape[0]:,} samples")
    logging.info(f"Test:  {X_test.shape[0]:,} samples")

    # Class distribution check
    logging.info("Calculating class distribution ...")
    for split_name, y_split in [("Train", y_train), ("Test", y_test)]:
        n_pos = y_split.sum()
        n_total = len(y_split)
        logging.info(f"{split_name}: {n_pos:,} defaults ({n_pos/n_total*100:.2f}%)")

    # Build and fit preprocessor
    logging.info("Building preprocessing pipeline ...")
    preprocessor = build_preprocessor(numeric_cols, low_card_cats, high_card_cats)
    logging.info("Fitting on training data ...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    logging.info("Transforming test data ...")
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out().tolist()
    # Clean up feature names (remove transformer prefixes)
    feature_names = [name.split("__", 1)[-1] if "__" in name else name for name in feature_names]
    logging.info(f"Total features after preprocessing: {len(feature_names)}")

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    # Save all artifacts
    logging.info("Saving Artifacts ...")

    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    logging.info(f"Dumped preprocessor.joblib")

    X_train_df.to_parquet(ARTIFACTS_DIR / "X_train.parquet", index=False)
    X_test_df.to_parquet(ARTIFACTS_DIR / "X_test.parquet", index=False)
    logging.info(f"Saving X_train.parquet ({X_train_df.shape})")
    logging.info(f"Saving X_test.parquet ({X_test_df.shape})")

    pd.DataFrame({"TARGET": y_train.values}).to_parquet(ARTIFACTS_DIR / "y_train.parquet", index=False)
    pd.DataFrame({"TARGET": y_test.values}).to_parquet(ARTIFACTS_DIR / "y_test.parquet", index=False)
    logging.info(f"Saving y_train.parquet, y_test.parquet")

    pd.DataFrame({"SK_ID_CURR": ids_train.values}).to_parquet(ARTIFACTS_DIR / "train_ids.parquet", index=False)
    pd.DataFrame({"SK_ID_CURR": ids_test.values}).to_parquet(ARTIFACTS_DIR / "test_ids.parquet", index=False)
    logging.info(f"Saving train_ids.parquet, test_ids.parquet")

    joblib.dump(feature_names, ARTIFACTS_DIR / "feature_names.joblib")
    logging.info(f"Saving feature_names.joblib ({len(feature_names)} features)")

    logging.info("Done.")


if __name__ == "__main__":
    main()
