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
    "Passenger_Count", "Base_Fare", "Per_Km_Rate", 
    "Per_Minute_Rate",
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

def build_pipeline(num, cat):
    transformers = []
    
    if num:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False))
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", _onehot()),
        ]), cat))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    
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
    stats= {}
    for c in num:
        s = X[c].astype(float)
        stats[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "p95": float(s.quantile(0.95)),
        }

    defaults_num = {c: float(pd.to_numeric(X[c], errors="coerce").median()) for c in num}
    defaults_cat = {}
    #for c in cat:
    #    mode = X[c].mode(dropna=True)
    #    if not mode.empty:
    #        defaults_cat[c] = str(mode.iloc[0])

    schema = {
        "feature_set": "all_raw",
        "model_name": "RandomForestRegressor",
        "numeric": num,
        "categorical": cat,
        "target": TARGET,
        "unit_currency": "EUR",
        "defaults": {**defaults_num, **defaults_cat},
        "stats": stats,
    }

    with open(SCHEMA_FILE, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


    print(f"saved model to {MODEL_FILE}")
    print(f"saved schema to {SCHEMA_FILE}")

