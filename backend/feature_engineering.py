import pandas as pd
from utils.setup import CLEAN_FILE, FEATURE_FILE, PROCESSED_DIR

TARGET = "Trip_Price"
NUMERIC = ["Trip_Distance_km", "Trip_Duration_Minutes",
           "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate"]
##CATEGORIC = ["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"]
CATEGORIC = []

def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_FILE)

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    if set(NUMERIC).issuperset({"Base_Fare", "Per_Km_Rate", "Per_Minute_Rate", "Trip_Distance_Km",
                                "Trip_Duration_Minutes"}):
        df["Rule_Estimate"] = (
            df["Base_Fare"]
            + df["Per_Km_Rate"] * df["Trip_Distance_Km"]
            + df["Per_Minute_Rate"] * df["Trip_Duration_Minutes"]
        )
        if TARGET in df.columns:
            df["Diff_to_Rule"] = (df[TARGET] - df["Rule_Estimate"]).fillna(0)
        else:
            df["Diff_to_Rule"] = 0.0

    if {"Trip_Distance_km", "Trip_Duration_Minutes"}.issubset(df.columns):
        df["Km_per_Min"] = (df["Trip_Distance_km"] / df["Trip_Duration_Minutes"]).replace([float("inf")], 0).fillna(0)

    return df

def save_features(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_FILE, index=False)

if __name__ == "__main__":
    df = load_clean()
    df = engineer(df)
    save_features(df)
    print(f"Saved features to {FEATURE_FILE} with {len(df)} rows and {len(df.columns)} columns.")