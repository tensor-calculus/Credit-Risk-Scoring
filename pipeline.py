from prefect import task, flow
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

@task(name="Load Data", retries=2, retry_delay_seconds=60)
def load_data():
    print("Running 01_load_data.py...")
    subprocess.run(["python", str(SRC_DIR / "01_load_data.py")], check=True)

@task(name="Feature Engineering", retries=2, retry_delay_seconds=60)
def feature_engineering():
    print("Running 02_feature_engineering.py...")
    subprocess.run(["python", str(SRC_DIR / "02_feature_engineering.py")], check=True)

@task(name="Preprocessing", retries=2, retry_delay_seconds=60)
def preprocessing():
    print("Running 03_preprocessing.py...")
    subprocess.run(["python", str(SRC_DIR / "03_preprocessing.py")], check=True)

@task(name="Train Model", retries=0) # Don't retry model training to save time
def train_model():
    print("Running 04_train_model.py...")
    subprocess.run(["python", str(SRC_DIR / "04_train_model.py")], check=True)

@task(name="Scorecard Generation", retries=1)
def generate_scorecard():
    print("Running 05_scorecard.py...")
    subprocess.run(["python", str(SRC_DIR / "05_scorecard.py")], check=True)

@task(name="Explainability", retries=1)
def generate_explainability():
    print("Running 06_explainability.py...")
    subprocess.run(["python", str(SRC_DIR / "06_explainability.py")], check=True)

@flow(name="Credit Risk ML Pipeline")
def run_pipeline():
    # Define execution DAG
    t1 = load_data()
    t2 = feature_engineering(wait_for=[t1])
    t3 = preprocessing(wait_for=[t2])
    t4 = train_model(wait_for=[t3])
    t5 = generate_scorecard(wait_for=[t4])
    t6 = generate_explainability(wait_for=[t5])

if __name__ == "__main__":
    run_pipeline()
