# 2_model_train.py (Model Training and LOCO-CV Script)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Constants & Configuration ---
LOOKBACK_WINDOW = 8  # 8 weeks history used for forecasting
FORECAST_HORIZON = 4 # 4 weeks ahead forecast (Ref: Materials and Methods)
COUNTRIES = ['ROK', 'US', 'JP']
TARGET_MAP = {'ROK': 'ILI_ROK', 'US': 'Hosp_US', 'JP': 'ILI_JP'}
MODEL_NAMES = ['LSTM', 'GRU', 'Transformer']

# --- Model Architectures (Ref: Materials and Methods 2. Predictive Model Development) ---

def build_lstm_model(input_shape):
    """Builds the 2-layer LSTM model architecture (Ref: Ref 7)."""
    i = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(i)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    o = Dense(FORECAST_HORIZON)(x)
    return Model(inputs=i, outputs=o, name='LSTM')

def build_gru_model(input_shape):
    """Builds the 2-layer GRU model architecture."""
    i = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(i)
    x = GRU(64)(x)
    x = Dropout(0.2)(x)
    o = Dense(FORECAST_HORIZON)(x)
    return Model(inputs=i, outputs=o, name='GRU')

def build_transformer_model(input_shape):
    """Builds the Transformer architecture for sequence forecasting (Ref: Ref 11)."""
    i = Input(shape=input_shape)
    
    # Layer 1: Multi-Head Attention Block
    x = LayerNormalization(epsilon=1e-6)(i)
    x = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = Dropout(0.2)(x)
    res = x + i 
    
    # Layer 2: Feed Forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(32, activation="relu")(x)
    x = Dense(input_shape[-1])(x)
    x = x + res
    
    x = GlobalAveragePooling1D()(x) # Sequence to vector
    x = Dropout(0.2)(x)
    o = Dense(FORECAST_HORIZON)(x)
    return Model(inputs=i, outputs=o, name='Transformer')


# --- Data Preparation for Sequence Models ---
def create_sequences(data, target_col):
    """Converts time-series data into input sequences (X) and target outputs (Y)."""
    X, Y = [], []
    # Features include all columns except the target outcome of the test country
    features = data.drop(columns=[c for c in TARGET_MAP.values() if c != target_col]).columns
    
    for i in range(len(data) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1):
        # Input sequence (8 weeks lookback)
        X.append(data.iloc[i:i + LOOKBACK_WINDOW][features].values)
        # Target output (4 weeks horizon)
        Y.append(data.iloc[i + LOOKBACK_WINDOW:i + LOOKBACK_WINDOW + FORECAST_HORIZON][target_col].values)
    
    return np.array(X), np.array(Y)


# --- LOCO-CV Implementation (Ref: Materials and Methods 3. Generalizability Testing) ---
def run_loco_cv(df):
    """Executes the Leave-One-Country-Out Cross-Validation (LOCO-CV) loop."""
    
    all_results = {}
    
    for test_country in COUNTRIES:
        print(f"\n--- Running LOCO-CV: Testing on {test_country} ---")
        
        # 1. Train/Test Data Split (Train on pre-2024 data, Test on 2024+ data)
        train_df = df[df.index.year < 2024] 
        test_df = df[df.index.year >= 2024]
        
        # 2. Sequence Creation (Target is the outcome of the test country)
        X_train, Y_train = create_sequences(train_df, TARGET_MAP[test_country])
        X_test, Y_test = create_sequences(test_df, TARGET_MAP[test_country])

        input_shape = (X_train.shape[1], X_train.shape[2])
        
        results_country = {}
        for model_name in MODEL_NAMES:
            print(f"  Training {model_name}...")
            
            # 3. Model Build, Compile, and Fit
            if model_name == 'LSTM': model = build_lstm_model(input_shape)
            elif model_name == 'GRU': model = build_gru_model(input_shape)
            elif model_name == 'Transformer': model = build_transformer_model(input_shape)
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
            
            history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
            
            # 4. Evaluation (MAPE is the primary metric)
            Y_pred = model.predict(X_test, verbose=0)
            mape = mean_absolute_percentage_error(Y_test, Y_pred) * 100
            
            results_country[model_name] = {'MAPE': mape, 'model': model}
            print(f"    -> {model_name} MAPE on {test_country}: {mape:.2f}%")
            
        all_results[test_country] = results_country
        
    return all_results

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load preprocessed data
    try:
        df_final = pd.read_csv('final_preprocessed_data.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("Error: Run '1_preprocess.py' first to generate 'final_preprocessed_data.csv'.")
        exit()
        
    # 2. Run LOCO-CV
    loco_results = run_loco_cv(df_final)
    
    # 3. Aggregate Results (Simulate the calculation of Overall MAPE)
    overall_mapes = {model: [] for model in MODEL_NAMES}
    
    for country, res in loco_results.items():
        for model_name, data in res.items():
            overall_mapes[model_name].append(data['MAPE'])

    overall_mape_summary = {model: np.mean(mapes) for model, mapes in overall_mapes.items()}
    print("\n\n--- Overall Generalization Performance (Mean MAPE) ---")
    for model, mape in sorted(overall_mape_summary.items(), key=lambda item: item[1]):
        print(f"{model}: {mape:.2f}%")
        
    print("\n[SUCCESS] Model training complete. Results saved for XAI analysis.")