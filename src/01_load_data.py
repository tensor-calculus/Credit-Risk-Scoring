"""
01_load_data.py - Loads Home Credit CSV tables into DuckDB
Reads all 8 raw CSV files from data/ and creates a persistent DuckDB
database at artifacts/credit_risk.duckdb.
"""

import duckdb
import logging
import os
from pathlib import Path

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_PATH = ARTIFACTS_DIR / "credit_risk.duckdb"

# Tables to load: (table_name, csv_filename)
TABLES = [
    ("application_train", "application_train.csv"),
    ("application_test", "application_test.csv"),
    ("bureau", "bureau.csv"),
    ("bureau_balance", "bureau_balance.csv"),
    ("previous_application", "previous_application.csv"),
    ("installments_payments", "installments_payments.csv"),
    ("pos_cash_balance", "pos_cash_balance.csv"),
    ("credit_card_balance", "credit_card_balance.csv"),
]

# # Indices to create: (table_name, column_name)
# INDICES = [
#     ("application_train", "SK_ID_CURR"),
#     ("application_test", "SK_ID_CURR"),
#     ("bureau", "SK_ID_CURR"),
#     ("bureau", "SK_ID_BUREAU"),
#     ("bureau_balance", "SK_ID_BUREAU"),
#     ("previous_application", "SK_ID_CURR"),
#     ("previous_application", "SK_ID_PREV"),
#     ("installments_payments", "SK_ID_CURR"),
#     ("installments_payments", "SK_ID_PREV"),
#     ("pos_cash_balance", "SK_ID_CURR"),
#     ("pos_cash_balance", "SK_ID_PREV"),
#     ("credit_card_balance", "SK_ID_CURR"),
#     ("credit_card_balance", "SK_ID_PREV"),
# ]


def load_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Load each CSV into a DuckDB table."""
    for table_name, csv_file in TABLES:
        csv_path = DATA_DIR / csv_file
        if not csv_path.exists():
            logging.warning(f"{csv_file} was skipped because it was not found in the given path")
            continue

        logging.info(f"Loading {csv_file} into {table_name} ...")
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(
            f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path.as_posix()}', header=true, sample_size=100_000)
            """
        )
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logging.info(f"Added {row_count:,} rows to {table_name}")


# def create_indices(con: duckdb.DuckDBPyConnection) -> None:
#     """Create indices on join columns for faster querying."""
#     print("\nCreating indices...")
#     for table_name, col_name in INDICES:
#         idx_name = f"idx_{table_name}_{col_name}"
#         try:
#             con.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}({col_name})")
#             print(f"  {idx_name}")
#         except Exception as e:
#             print(f"  [WARN] {idx_name}: {e}")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Remove existing DB for a clean load
    if DB_PATH.exists():
        DB_PATH.unlink()
        logging.info(f"Removed existing database: {DB_PATH}")

    logging.info(f"Creating DuckDB database: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    logging.info("Loading tables ...")
    load_tables(con)

    # create_indices(con)

    # Summary
    logging.info("Database summary: ")
    tables = con.execute("SHOW TABLES").fetchall()
    for (t,) in tables:
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols = con.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name='{t}'").fetchone()[0]
        logging.info(f"    {t:30s} {cnt:>12,} rows {cols:>4} cols")

    con.close()
    logging.info(f"Done. Database saved to {DB_PATH}")


if __name__ == "__main__":
    main()
