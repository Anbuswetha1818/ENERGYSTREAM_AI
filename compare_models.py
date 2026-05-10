import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from preprocessing import load_data, preprocess_data
from model import build_lstm_model
import tensorflow as tf

def invert_scaling(data, scaler, num_features, target_idx):
    dummy = np.zeros((len(data), num_features))
    dummy[:, target_idx] = data.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]

def main():
    filepath = 'household_power_consumption.txt'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    # 1. Load and Preprocess
    df = load_data(filepath)
    X_lstm, y_lstm, scaler, cols = preprocess_data(df)
    
    # Split for LSTM
    train_size = int(len(X_lstm) * 0.8)
    X_lstm_train, X_lstm_test = X_lstm[:train_size], X_lstm[train_size:]
    y_lstm_train, y_lstm_test = y_lstm[:train_size], y_lstm[train_size:]
    
    # 2. Prepare data for Traditional Models (Flatten X)
    # Traditional models don't take sequences directly in the same way, 
    # but we can flatten the window to use the same information.
    X_trad_train = X_lstm_train.reshape(X_lstm_train.shape[0], -1)
    X_trad_test = X_lstm_test.reshape(X_lstm_test.shape[0], -1)
    y_trad_train = y_lstm_train.flatten()
    y_trad_test = y_lstm_test.flatten()

    results = {}

    # --- Decision Tree ---
    print("\nTraining Decision Tree Regressor...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_trad_train, y_trad_train)
    y_pred_dt = dt.predict(X_trad_test)
    
    # --- Random Forest ---
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_trad_train, y_trad_train)
    y_pred_rf = rf.predict(X_trad_test)

    # --- LSTM (Load or Re-train) ---
    # Since we want a fair comparison on the same split:
    print("Training LSTM Model for comparison...")
    model = build_lstm_model((X_lstm_train.shape[1], X_lstm_train.shape[2]))
    model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=72, verbose=0, shuffle=False)
    y_pred_lstm = model.predict(X_lstm_test)

    # --- Evaluation ---
    num_features = len(cols)
    target_idx = list(cols).index('Global_active_power')

    def get_metrics(y_true_norm, y_pred_norm, label):
        y_true_inv = invert_scaling(y_true_norm, scaler, num_features, target_idx)
        y_pred_inv = invert_scaling(y_pred_norm, scaler, num_features, target_idx)
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        print(f"{label} RMSE: {rmse:.6f} kW")
        return rmse

    results['Decision Tree'] = get_metrics(y_trad_test, y_pred_dt, "Decision Tree")
    results['Random Forest'] = get_metrics(y_trad_test, y_pred_rf, "Random Forest")
    results['LSTM'] = get_metrics(y_lstm_test, y_pred_lstm, "LSTM")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    rmses = list(results.values())
    
    bars = plt.bar(models, rmses, color=['blue', 'green', 'orange'])
    plt.ylabel('RMSE (kW)')
    plt.title('Model Performance Comparison (Load Forecasting)')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

    plt.savefig('model_comparison.png')
    print("\nComparison plot saved to 'model_comparison.png'")

if __name__ == "__main__":
    main()
