"""
02_feature_engineering.py - Executes SQL feature engineering scripts.
Runs the 5 SQL aggregation queries on DuckDB, joins them onto the main
application table and also handles sentinel values.
"""

import duckdb
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_PATH = ARTIFACTS_DIR / "credit_risk.duckdb"

# SQL files to execute in order: (table_name, sql_filename)
FEATURE_QUERIES = [
    ("bureau_features",       "bureau_features.sql"),
    ("previous_app_features", "previous_app_features.sql"),
    ("installments_features", "installments_features.sql"),
    ("pos_cash_features",     "pos_cash_features.sql"),
    ("credit_card_features",  "credit_card_features.sql"),
]

# Sentinel value used by Home Credit for "infinity" / missing in DAYS columns
SENTINEL_VALUE = 365243

def execute_feature_queries(con: duckdb.DuckDBPyConnection) -> None:
    """Read each SQL file and create a table from the query result."""
    for table_name, sql_file in FEATURE_QUERIES:
        sql_path = SQL_DIR / sql_file
        logging.info(f"Executing {sql_file} -> {table_name} ...")
        sql = sql_path.read_text(encoding="utf-8")
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS ({sql})")
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logging.info(f"Added {row_count:,} rows")


def build_final_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Execute the join query and return the final DataFrame."""
    join_sql_path = SQL_DIR / "join_all_features.sql"
    logging.info(f"Executing join_all_features.sql ...")
    join_sql = join_sql_path.read_text(encoding="utf-8")
    df = con.execute(join_sql).fetchdf()
    logging.info(f"Final dataset shape: {df.shape}")
    return df


def handle_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel value 365243 with NaN in DAYS related columns."""
    days_cols = [c for c in df.columns if "DAYS_" in c]
    replaced_count = 0
    for col in days_cols:
        mask = df[col] == SENTINEL_VALUE
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            replaced_count += n
            logging.info(f"In {col}: replaced {n:,} sentinel values with NaN")
    logging.info(f"Total sentinel replacements: {replaced_count:,}")
    return df


def print_missingness_report(df: pd.DataFrame) -> None:
    """Print missingness assessment for columns with missing values."""
    print("\n" + "=" * 70)
    print("Missingness Report")
    print("=" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_cols = missing[missing > 0].sort_values(ascending=False)

    if len(missing_cols) == 0:
        print("  No missing values found.")
        return

    print(f"{'Column':<45} {'Missing':>10} {'Pct':>8}  Type")
    print("-" * 80)

    # Missingness type classification:
    # - MCAR (Missing Completely at Random): housing features - missing for non-homeowners
    # - MAR  (Missing at Random): bureau/prev features - missing because client had no prior credit
    # - MNAR (Missing Not at Random): OWN_CAR_AGE - missing because they don't own a car (value itself determines missingness)
    for col in missing_cols.index[:50]:  # Show top 50
        n = missing_cols[col]
        pct = missing_pct[col]

        # Classify missingness type
        if col.startswith("BUREAU_") or col.startswith("PREV_") or col.startswith("INST_") or \
           col.startswith("POS_") or col.startswith("CC_"):
            mtype = "MAR"   # Missing because no prior loans/bureau records
        elif "APARTMENT" in col or "BASEMENT" in col or "LIVING" in col or \
             "FLOOR" in col or "LAND" in col or "NONLIVING" in col or \
             "COMMONAREA" in col or "ELEVATOR" in col or "ENTRANCE" in col or \
             "FONDKAPREMONT" in col or "HOUSETYPE" in col or "WALLSMATERIAL" in col or \
             "EMERGENCYSTATE" in col or "TOTALAREA" in col or "YEARS_BUILD" in col or \
             "YEARS_BEGINEXPLUATATION" in col:
            mtype = "MCAR"  # Housing info missing for non-homeowners
        elif col == "OWN_CAR_AGE":
            mtype = "MNAR"  # Missing because no car owned
        elif col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
            mtype = "MAR"   # External score not available for all applicants
        elif col == "OCCUPATION_TYPE":
            mtype = "MAR"   # Occupation not reported
        else:
            mtype = "MAR"   # Default assumption

        print(f"  {col:<43} {n:>10,} {pct:>7.1f}%  {mtype}")


def main() -> None:
    print(f"Connecting to DuckDB: {DB_PATH}\n")
    con = duckdb.connect(str(DB_PATH))

    print("=" * 60)
    print("Executing Feature Engineering SQL Queries")
    print("=" * 60)
    execute_feature_queries(con)

    print("\n" + "=" * 60)
    print("Building Final Feature Matrix")
    print("=" * 60)
    df = build_final_dataset(con)

    print("\n" + "=" * 60)
    print("Handling Sentinel Values (365243 -> NaN)")
    print("=" * 60)
    df = handle_sentinel_values(df)

    print_missingness_report(df)

    # Save to parquet
    output_path = ARTIFACTS_DIR / "features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nFeature matrix saved to {output_path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Target Distribution")
    print("=" * 60)
    target_counts = df["TARGET"].value_counts()
    total = len(df)
    for val, cnt in target_counts.items():
        print(f"  TARGET={val}: {cnt:>8,}  ({cnt/total*100:.1f}%)")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
