from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import json
import joblib
import pandas as pd
from utils.setup import MODEL_FILE, SCHEMA_FILE

app = FastAPI(title="Taxi Prediction API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MODEL = None
SCHEMA: Dict[str, Any] = {}

@app.on_event("startup")
def _load_artifacts():
    global MODEL, SCHEMA
    try:
        MODEL = joblib.load(MODEL_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_FILE}: {e}")
    try:
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            SCHEMA = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load schema from {SCHEMA_FILE}: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "model_loaded": MODEL is not None,
        "features": {
            "numeric": SCHEMA.get("numeric", []),
            "categoric": SCHEMA.get("categoric", [])
        }
    }

@app.get("/schema")
def get_schema():
    if not SCHEMA:
        raise HTTPException(status_code=500, detail="Schema not loaded.")
    return SCHEMA

def _validate_and_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    if not SCHEMA:
        raise HTTPException("Schema not loaded.")
    required = list(SCHEMA.get("numeric", [])) + list(SCHEMA.get("categoric", []))
    missing = [c for c in required if c not in payload]
    if missing:
        raise HTTPException(status_code=400, detail={"error": "Missing features", "missing": missing})
    
    row: Dict[str, Any] = {c: payload.get(c) for c in required}

    for c in SCHEMA.get("numeric", []):
        v = row.get(c)
        if v is not None:
            try:
                row[c] = float(v)
            except Exception:
                raise HTTPException(status_code=400, detail =f"column '{c}' has to be numerical, got: {v!r}")
            for c in SCHEMA.get("categoric", []):
                v = row.get(c)
                if v is not None and not isinstance(v, str):
                    row[c] = str(v)
    
    return pd.DataFrame([row])

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    Send a JSON with the exact features that are located within /schema.
    Exempel:
    {
      "Trip_Distance_km": 8, "Trip_Duration_Minutes": 15, "Passenger_Count": 1,
      "Base_Fare": 40, "Per_Km_Rate": 12, "Per_Minute_Rate": 3,
      "Time_of_Day": "Evening", "Day_of_Week": "Friday",
      "Traffic_Conditions": "Moderate", "Weather": "Clear",
      "Rule_Estimate": 40 + 12*8 + 3*15, "Diff_to_Rule": 0, "Km_per_Min": 8/15
    }
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    df = _validate_and_frame(payload)
    try: 
        pred = MODEL.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    return {"predicted_price": float(pred)}
