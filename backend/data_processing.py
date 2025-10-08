import pandas as pd
from utils.setup import RAW_FILE, PROCESSED_DIR, CLEAN_FILE, ensure_dirs

ensure_dirs()

NUM_COLS = [
    "Trip_Distance_km",  "Trip_Duration_Minutes", "Trip_Price",
    "Passenger_Count", "Base_Fare", "Per_Km_Rate", "Per_Minute_Rate",
]
CAT_COLS = ["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"]


def clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing categorical and passenger values with sensible defaults.
    
    Args:
        df: DataFrame with missing categorical values
        
    Returns:
        DataFrame with filled categorical features
    """
    df = df.copy()
    total_filled = 0
    
    categorical_defaults = {
        'Time_of_Day': 'Afternoon',
        'Day_of_Week': 'Weekday',
        'Traffic_Conditions': 'Medium',
        'Weather': 'Clear'
    }
    
    for col, default in categorical_defaults.items():
        missing = df[col].isnull().sum()
        if missing > 0:
            fill_value = df[col].mode()[0] if not df[col].mode().empty else default
            df[col] = df[col].fillna(fill_value)
            total_filled += missing
    
    # Fill passenger count with median
    missing_passengers = df['Passenger_Count'].isnull().sum()
    if missing_passengers > 0:
        df['Passenger_Count'] = df['Passenger_Count'].fillna(1)
        total_filled += missing_passengers
      
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for improved model performance.
    
    Generates weather/traffic multipliers, time-based indicators, 
    and interaction features that capture pricing patterns.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    print("Creating engineered features...")
    
    # Weather and traffic impact multipliers
    df['Weather_Impact'] = df['Weather'].map({'Clear': 1.0, 'Rain': 1.15, 'Snow': 1.3})
    df['Traffic_Multiplier'] = df['Traffic_Conditions'].map({'Low': 1.0, 'Medium': 1.1, 'High': 1.25})
    
    # Time-based binary features
    df['is_morning_rush'] = (df['Time_of_Day'] == 'Morning').astype(int)
    df['is_evening_rush'] = (df['Time_of_Day'] == 'Evening').astype(int)
    df['is_peak_hours'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    df['is_weekend'] = (df['Day_of_Week'] == 'Weekend').astype(int)
    
    # High-impact trip indicator
    df['high_impact_trip'] = ((df['Weather'] == 'Snow') | (df['Traffic_Conditions'] == 'High')).astype(int)
    
    # Interaction features
    df['condition_score'] = df['Weather_Impact'] * df['Traffic_Multiplier']
    df['distance_x_conditions'] = df['Trip_Distance_km'] * df['condition_score']
    
    print("Added 8 engineered features")
    df = df.drop(columns=["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"])
    return df

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILE)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates()

    for c in ["Trip_Distance_km", "Trip_Duration_Minutes", "Trip_Price"]:
        if c in df.columns:
            df = df[df[c].isna() | (df[c] >= 0)]

    for c in ["Trip_Distance_km", "Trip_Duration_Minutes", 
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
    df_filled = clean_categorical_features(df_raw)
    df_clean = clean(df_filled)
    df_altered = engineer_features(df_clean)
    print(f"Cleaned shape: {df_altered.shape}")
    split_and_save(df_altered)