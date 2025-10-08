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
    return {"numeric": SCHEMA.get("numeric", []),
        "categorical": SCHEMA.get("categorical", []),
        "target": SCHEMA.get("target"),
        "unit_currency": SCHEMA.get("unit_currency", "EUR"),
        "defaults": SCHEMA.get("defaults", {}),
        "stats": SCHEMA.get("stats", {}),
    }

def _validate_and_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    if not SCHEMA:
        raise HTTPException("Schema not loaded.")
    numeric = list(SCHEMA.get("numeric", [])) 
    categoric = list(SCHEMA.get("categoric", []))
    defaults = SCHEMA.get("defaults", {}) or {}
    
    row: Dict[str, Any] = {}

    for c in numeric + categoric:
        if c in payload and payload[c] is not None:
            row[c] = payload[c]
        elif c in defaults:
            row[c] = defaults[c]
        else:
            # om vi inte har default, lämna som missing (fel om numeric)
            row[c] = None

    # kasta numeriska
    for c in numeric:
        v = row.get(c)
        if v is None:
            raise HTTPException(status_code=400, detail=f"Missing required numeric feature: {c}")
        try:
            row[c] = float(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Feature '{c}' must be numeric, got {v!r}")

    # kasta kategoriska
    for c in categoric:
        v = row.get(c)
        if v is None:
            # försök fallback igen
            v = defaults.get(c, "")
        if v is not None and not isinstance(v, str):
            v = str(v)
        row[c] = v

    return pd.DataFrame([row])


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try: 
        df = _validate_and_frame(payload)
        pred = MODEL.predict(df)[0]
        return {"predicted_price": float(pred)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
