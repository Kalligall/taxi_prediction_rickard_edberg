import pandas as pd
from utils.setup import RAW_FILE, PROCESSED_DIR, CLEAN_FILE, ensure_dirs

ensure_dirs()

NUM_COLS = [
    "Trip_Distance_km", "Passenger_Count", "Base_Fare",
    "per_Km_Rate", "Per_Minute_Rate", "Trip_Duration_Minutes", "Trip_price"
]
CAT_COLS = ["Time_of_Day", "Day_of_Week", "Month", "Traffic_Conditions", "Weather"]

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILE)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates()

    for c in ["Trip_Distance_km", "Trip_Duraction_Minutes", "Passenger_Count", "Trip_Price"]:
        if c in df.columns:
            df = df[df[c].isna() | (df[c] >= 0)]

    for c in ["Trip_Distance_km", "Trip_Duration_Minutes", "Passenger_Count", 
              "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    for c in CAT_COLS:
        if c in df.columns:
            mode = df[c].mode(dropna=True)
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])

    if "Trip_Price" in df.columns:
        cap = df["Trip_Price"].quantile(0.99)
        df = df[df["Trip_Price"].isna() | (df["Trip_Price"] <= cap)]

    return df

def split_and_save(df: pd.DataFrame) -> None:
    """Save:
       - taxi_clean.csv
       - taxi_unlabeled.csv
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    labeled = df.dropna(subset=["Trip_Price"])
    unlabeled = df[df["Trip_Price"].isna()]

    labeled.to_csv(CLEAN_FILE, index=False)
    (PROCESSED_DIR / "taxi_unlabeled.csv").write_text("") if unlabeled.empty \
        else unlabeled.to_csv(PROCESSED_DIR / "taxi_unlabeled.csv", index=False)
    
    print(f"Saved: {CLEAN_FILE} (labeled={len(labeled)} lines)")
    print(f"Saved: {PROCESSED_DIR/"taxi_unlabeled.csv"} (unlabeled ={len(unlabeled)} lines)")

if __name__ == "__main__":
    print(f"Reading raw data from: {RAW_FILE}")
    df_raw = load_raw()
    print(f"Raw shape: {df_raw.shape}")
    df_clean = clean(df_raw)
    print(f"Cleaned shape: {df_clean.shape}")
    split_and_save(df_clean)