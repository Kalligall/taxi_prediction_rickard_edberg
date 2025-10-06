from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

RAW_FILE = RAW_DIR / "taxi_trip_pricing.csv"
CLEAN_FILE = PROCESSED_DIR / "taxi_clean.csv"
FEATURES_FILE = PROCESSED_DIR / "taxi_features.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
SCHEMA_FILE = MODELS_DIR / "schema.json"

def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()
    print(f"Ensured directories:\n- {ROOT} \n- {RAW_DIR}\n- {PROCESSED_DIR}\n- {MODELS_DIR}")