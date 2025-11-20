# 1_preprocess.py (Data Preprocessing Script)

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# --- 1. Load Data ---
def load_data(file_path='ILI_data_aggregated.csv'):
    """Loads epidemiological, weather, and search trend data."""
    try:
        # Assumption: data includes 'Date' index and columns like 'ILI_ROK', 'Temp_ROK', etc.
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please ensure data is correctly placed.")
        return pd.DataFrame()
    return df

# --- 2. Data Cleaning and Imputation (Ref: Materials and Methods 1. Data Preprocessing) ---
def handle_missing_data(df):
    """Handles missing values using methods specified in the study."""
    # 1. Epidemiological data (ILI/Hosp): Apply linear interpolation
    epi_cols = [col for col in df.columns if 'ILI' in col or 'Hosp' in col]
    df[epi_cols] = df[epi_cols].interpolate(method='linear')

    # 2. Meteorological data: Replace with corresponding monthly average
    weather_cols = [col for col in df.columns if 'Temp' in col or 'Humidity' in col]
    for col in weather_cols:
        # Fill NaN values with the mean of the corresponding month
        df[col] = df[col].fillna(df[col].groupby(df.index.month).transform('mean'))
    
    return df

# --- 3. Normalization (Ref: Ref 8 - Z-score) ---
def normalize_data(df):
    """Performs Z-score normalization on all feature columns."""
    df_normalized = df.copy()
    
    for col in df_normalized.columns:
        mean = df_normalized[col].mean()
        std = df_normalized[col].std()
        # Avoid division by zero
        if std != 0:
            df_normalized[col] = (df_normalized[col] - mean) / std
        
    return df_normalized

# --- 4. Stationarity Check and Differencing (Ref: Ref 9 - ADF Test) ---
def make_stationary(df, outcome_cols):
    """Performs ADF test and applies first-order differencing if non-stationary."""
    df_diff = df.copy()
    
    for col in outcome_cols:
        if df[col].empty or len(df[col].dropna()) < 10:
            continue

        result = adfuller(df[col].dropna())
        p_value = result[1]
        
        # If p-value > 0.05, series is non-stationary; apply first-order differencing.
        if p_value > 0.05:
            print(f"Applying differencing: {col} (p={p_value:.4f})")
            df_diff[col] = df[col].diff(1)
        
    return df_diff.dropna()

# --- Main Execution ---
if __name__ == '__main__':
    raw_df = load_data()
    if raw_df.empty:
        exit()
        
    cleaned_df = handle_missing_data(raw_df)
    normalized_df = normalize_data(cleaned_df)
    
    # Target outcome variables (ILI/Hosp)
    outcome_variables = [col for col in normalized_df.columns if 'ILI' in col or 'Hosp' in col]
    final_data = make_stationary(normalized_df, outcome_variables)
    
    print("\n[SUCCESS] Preprocessing complete. Final dataset structure:")
    print(final_data.head())
    
    # Save for the next modeling script
    final_data.to_csv('final_preprocessed_data.csv')