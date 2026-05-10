import os
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import load_data, preprocess_data
from model import build_lstm_model
from sklearn.metrics import mean_squared_error
def main():python
    filepath = 'household_power_consumption.txt'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please ensure the dataset is in the project directory.")
        return

    # 1. Load and Preprocess
    df = load_data(filepath)
    X, y, scaler, cols = preprocess_data(df)
    
    # Split into train and test (80-20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # 2. Build and Train Model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    print("Starting model training (this may take a few minutes)...")
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=72, 
        validation_data=(X_test, y_test), 
        verbose=1, 
        shuffle=False
    )

    # 3. Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate MSE and RMSE on normalized data
    mse_norm = mean_squared_error(y_test, y_pred)
    rmse_norm = np.sqrt(mse_norm)
    print(f"Normalized MSE: {mse_norm:.6f}")
    print(f"Normalized RMSE: {rmse_norm:.6f}")

    # Inverse scaling for metrics in original units
    num_features = len(cols)
    target_idx = list(cols).index('Global_active_power')
    
    def invert_scaling(data, scaler, num_features, target_idx):
        dummy = np.zeros((len(data), num_features))
        dummy[:, target_idx] = data.flatten()
        inv = scaler.inverse_transform(dummy)
        return inv[:, target_idx]

    y_test_inv = invert_scaling(y_test, scaler, num_features, target_idx)
    y_pred_inv = invert_scaling(y_pred, scaler, num_features, target_idx)

    # Calculate MSE and RMSE in original units (kW)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE) in original units: {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE) in original units: {rmse:.6f}")

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv[:200], label='Actual Global Active Power (kW)')
    plt.plot(y_pred_inv[:200], label='Predicted Global Active Power (kW)')
    plt.title('Energy Consumption Forecasting - LSTM (Sample of 200 hours)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()
    plt.savefig('prediction_plot.png')
    print("Saved prediction plot to 'prediction_plot.png'")
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    print("Saved loss plot to 'loss_plot.png'")

if __name__ == "__main__":
    main()
