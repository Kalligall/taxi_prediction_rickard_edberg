import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from utils.setup import FEATURE_FILE, MODEL_FILE, SCHEMA_FILE, MODELS_DIR

TARGET = "Trip_Price"
NUMERIC = [
    "Trip_Distance_km", "Trip_Duration_Minutes",
    "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate",
    "Rule_Estimate", "Diff_to_Rule", "Km_per_Min"
]
##CATEGORIC = ["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"]
CATEGORIC = []

def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

try:
    from sklearn.metrics import root_mean_squared_error as _rmse
    def RMSE(y_true, y_pred) -> float:
        return float(_rmse(y_true, y_pred))
except Exception:
    def RMSE(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_errror(y_true, y_pred)))

def load_data():
    df = pd.read_csv(FEATURE_FILE).dropna(subset=[TARGET])
    X_cols = [c for c in NUMERIC + CATEGORIC if c in df.columns]
    X, y = df[X_cols].copy(), df[TARGET].copy()
    num = [c for c in NUMERIC if c in X.columns]
    cat = [c for c in CATEGORIC if c in X.columns]
    return X, y, num, cat

def build_pipeline(num_features, cat_features):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False))
            ]), num_features),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_features),
        ],
        remainder="drop"
    )
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    return Pipeline([("pre", pre), ("model", model)])

def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": RMSE(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred))
    }

if __name__ == "__main__":
    X, y, num, cat = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline(num, cat)
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    metrics = evaluate(yte, yhat)
    print(f"Baseline metrics: {metrics}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_FILE)

    schema = {"numeric": num, "categoric": cat, "target": TARGET}
    with open(SCHEMA_FILE, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"saved model to {MODEL_FILE}")
    print(f"saved schema to {SCHEMA_FILE}")
