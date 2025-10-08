import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from utils.setup import FEATURE_FILE

TARGET = "Trip_Price"
NUMERIC = [
    "Trip_Distance_km", "Trip_Duration_Minutes",
    "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate",
    "Rule_Estimate", "Diff_to_Rule", "Km_per_Min"
]
##CATEGORIC = ["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"]
CATEGORIC = []

REPORTS_DIR = Path("reports")
RESULTS_CSV = REPORTS_DIR / "model_results.csv"
SPLIT_FILE = REPORTS_DIR / "split_indices.json"
PRED_TPL = "predictions_{name}.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
def load_data():
    df = pd.read_csv(FEATURE_FILE).dropna(subset=[TARGET])
    X_cols = [c for c in NUMERIC + CATEGORIC if c in df.columns]
    X, y = df[X_cols].copy(), df[TARGET].copy()
    num = [c for c in NUMERIC if c in X.columns]
    cat = [c for c in CATEGORIC if c in X.columns]
    return X, y, num, cat

def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False))
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("onehot", _onehot())
            ]), cat),
        ],
        remainder="drop"
    )
    
def available_models() -> Dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42)
    }

def get_split_indices(n: int):
    if SPLIT_FILE.exists():
        idx = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        return np.array(idx["train"], dtype=int), np.array(idx["test"], dtype=int)
    else:
        all_idx = np.arange(n)
        tr,te = train_test_split(all_idx, test_size=0.2, random_state=42)
        SPLIT_FILE.write_text(json.dumps({"train": tr.tolist(), "test": te.tolist()}, indent=2), encoding="utf-8")
        return tr, te

def run_one(model_name: str):
    models = available_models()
    if model_name not in models:
        raise SystemExit(f"Unknown model: {model_name}. Use --list to see options.")
    
    X, y, num, cat = load_data()
    tr_idx, te_idx = get_split_indices(len(X))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

    pre = make_preprocessor(num, cat)
    mdl = models[model_name]
    pipe = Pipeline([("pre", pre), ("model", mdl)])

    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    yhat = pipe.predict(Xte)
    pred_s = time.perf_counter() - t1

    row = {
        "model": model_name,
        "MAE": float(mean_absolute_error(yte, yhat)),
        "RMSE": RMSE(yte, yhat),
        "R2": float(r2_score(yte, yhat)),
        "fit_sec": round(fit_s, 3),
        "pred_sec": round(pred_s, 3),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }

    if RESULTS_CSV.exists():
        df_res = pd.read_csv(RESULTS_CSV)
        df_res = df_res[df_res.model != model_name]
        df_res = pd.concat([df_res, pd.DataFrame([row])], ignore_index=True)
    else:
        df_res = pd.DataFrame([row])

    df_res = df_res.sort_values(["RMSE", "MAE"], ascending=[True, True])
    df_res.to_csv(RESULTS_CSV, index=False)
    print("Saved results to ->", RESULTS_CSV)
    print(df_res.to_string(index=False))

    preds = pd.DataFrame({
        "y_true": yte.values,
        "y_pred": yhat
    }, index=Xte.index)
    preds.to_csv(REPORTS_DIR / PRED_TPL.format(name=model_name), index_label="row_index")
    print("Saved predictions to ->", REPORTS_DIR / PRED_TPL.format(name=model_name))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run one model and save metrics/predictions.")
    ap.add_argument("--list", action="store_true", help="List available models.")
    ap.add_argument("--model", type=str, help="Model name to run.")
    args = ap.parse_args()

    if args.list:
        print("Available models:")
        for n in available_models().keys():
            print(" -", n)
        raise SystemExit(0)
    
    if not args.model:
        raise SystemExit("Please provide --model <name>. Try --list for options")
    
    run_one(args.model)