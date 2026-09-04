import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

app = FastAPI(title="Credit Risk Scoring API")

class ApplicantData(BaseModel):
    features: Dict[str, Any]

model = None
feature_names = None

@app.on_event("startup")
def load_artifacts():
    global model, feature_names
    try:
        if (ARTIFACTS_DIR / "model.joblib").exists():
            model = joblib.load(ARTIFACTS_DIR / "model.joblib")
        if (ARTIFACTS_DIR / "feature_names.joblib").exists():
            feature_names = joblib.load(ARTIFACTS_DIR / "feature_names.joblib")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

@app.get("/health")
def health_check():
    if model is None or feature_names is None:
        return {"status": "unhealthy", "message": "Model artifacts not loaded."}
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: ApplicantData):
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data.features])
        
        # Ensure all expected features are present
        missing_cols = set(feature_names) - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan
            
        # Reorder to match training
        df = df[feature_names]
        
        # Enforce numeric types for all columns to prevent LightGBM object errors
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Predict
        probability = model.predict_proba(df.values)[0, 1]
        
        return {
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
