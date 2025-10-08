import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
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
from utils.setup import CLEAN_FILE

REPORTS_DIR = Path("reports")
RESULTS_CSV = REPORTS_DIR / "model_results.csv"
SPLIT_FILE = REPORTS_DIR / "split_indices.json"

TARGET = "Trip_Price"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def RMSE(y_true, y_pred) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error as _rmse
        return float(_rmse(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
def available_models() -> Dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42)
    }

def feature_sets(df: pd.DataFrame) -> Dict[str, tuple[List[str], List[str]]]:
    all_cols = set(df.columns)

    def present(cols): return [c for c in cols if c in all_cols]

    dist_dur_num = present(["Trip_Distance_km", "Trip_Duration_Minutes"])

    dist_dur_fe = present(["Trip_Distance_km", "Trip_Duration_Minutes", "is_weekend", 
                          "is_morning_rush", "is_evening_rush", "is_peak_hours",
                          "condition_score", "Weather_Impact", "Traffic_Multiplier",
                          "high_impact_trip", "distance_x_conditions"
                          ])

    dist_dur_tariff_num = present([
        "Trip_Distance_km", "Trip_Duration_Minutes",
        "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate"
    ])

    raw_exclude = {TARGET}
    raw_cols = [c for c in df.columns if c not in raw_exclude]
    num_all = [c for c in raw_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_all = [c for c in raw_cols if pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]

    return {
        "dist_dur": (dist_dur_num, []),
        "dist_dur_tariff_num": (dist_dur_tariff_num, []),
        "all_raw": (num_all, cat_all),
        "dist_dur_fe": (dist_dur_fe, [])
    }



    
def load_data():
    df = pd.read_csv(CLEAN_FILE).copy()
    if TARGET not in df.columns:
        raise SystemExit(f"Target '{TARGET}' not in {CLEAN_FILE}. Columns: {list(df.columns)}")
    df = df.dropna(subset=[TARGET])
    return df

def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    transformers=[]
    if num:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False)),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", _onehot()),
        ]), cat))
    if not transformers:
        raise SystemExit("No features selected. Check your feature_set and data columns")
    return ColumnTransformer(transformers=transformers, remainder="drop")
    

def get_split_indices(n: int):
    if SPLIT_FILE.exists():
        idx = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        if "train" not in idx or "test" not in idx:
            raise ValueError(f"Invalid split file {SPLIT_FILE}: expected keys 'train' and 'test'.")
        return np.array(idx["train"], dtype=int), np.array(idx["test"], dtype=int)
    else:
        all_idx = np.arange(n)
        tr,te = train_test_split(all_idx, test_size=0.2, random_state=42)
        SPLIT_FILE.write_text(json.dumps({"train": tr.tolist(), "test": te.tolist()}, indent=2), encoding="utf-8")
        return tr, te

def run_one(model_name: str, feat_name: str):
    models = available_models()

    if model_name.upper() == "ALL":
        names_models = list(models.keys())
    else:
        if model_name not in models:
            raise SystemExit(f"Unknown model: {model_name}. Use --list to see options.")
        names_models = [model_name]
    
    df = load_data()
    sets = feature_sets(df)

    if feat_name.upper() == "ALL":
        names_feats = list(sets.keys())
    else:
        if feat_name not in sets:
            raise SystemExit(f"Unknown model: {model_name}. Use --list to see options.")
        names_feats = [feat_name]

    tr_idx, te_idx = get_split_indices(len(df))
    
    for f_name in names_feats:
        num, cat = sets[f_name]
        X = df[num + cat].copy()
        y = df[TARGET].copy()
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        pre = make_preprocessor(num, cat)
        
        for m_name in names_models:
            mdl = models[m_name]
            pipe = Pipeline([("pre", pre), ("model", mdl)])

            t0 = time.perf_counter()
            pipe.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            yhat = pipe.predict(Xte)
            pred_s = time.perf_counter() - t1

            row = {
                "feature_set": f_name,
                "model": m_name,
                "MAE": float(mean_absolute_error(yte, yhat)),
                "RMSE": RMSE(yte, yhat),
                "R2": float(r2_score(yte, yhat)),
                "fit_sec": round(fit_s, 3),
                "pred_sec": round(pred_s, 3),
                "n_train": int(len(Xtr)),
                "n_test": int(len(Xte)),
                "num_feats": len(num),
                "cat_feats": len(cat),
            }

            if RESULTS_CSV.exists():
                df_res = pd.read_csv(RESULTS_CSV)
                df_res = df_res[~((df_res["feature_set"] == f_name) & (df_res["model"] == m_name))]
                df_res = pd.concat([df_res, pd.DataFrame([row])], ignore_index=True)
            else:
                df_res = pd.DataFrame([row])

            df_res = df_res.sort_values(["RMSE", "MAE"], ascending=[True, True])
            df_res.to_csv(RESULTS_CSV, index=False)
            print("Saved results to ->", RESULTS_CSV)
            print(df_res.to_string(index=False))

            pred_path = REPORTS_DIR / f"predictions_{f_name}_{m_name}.csv"
            pd.DataFrame({"y_true": yte.values, "y_pred": yhat}, index=Xte.index).to_csv(pred_path, index_label="row_index")
            print("Saved predictions ->", pred_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run model with chosen feature sets and save metrics/predictions.")
    ap.add_argument("--list", action="store_true", help="List available models and feature sets.")
    ap.add_argument("--model", type=str, default="GradientBoosting", help="Model name to run or ALL.")
    ap.add_argument("--features", type=str, default="dist_dur", help="Feature set name or ALL")
    args = ap.parse_args()

    if args.list:
        df_tmp = load_data()
        sets = feature_sets(df_tmp)
        print("Available models:")
        for n in available_models().keys():
            print(" -", n)
        print("Available feature sets:")
        for k, (num, cat) in sets.items():
            print(f" - {k}: num={num} cat={cat}")
        raise SystemExit(0)
    
    if not args.model:
        raise SystemExit("Please provide --model <name> and feature <name> Try --list for options or ALL")
    
    run_one(args.model, args.features)