import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(filepath):
    """Loads the household power consumption dataset."""
    print(f"Loading data from {filepath}...")
    # Use semicolon delimiter as observed in the file
    df = pd.read_csv(filepath, sep=';', 
                     parse_dates={'dt': ['Date', 'Time']}, 
                     infer_datetime_format=True, 
                     low_memory=False, 
                     na_values=['?'], 
                     index_col='dt')

    
    # Fill missing values with the mean of the column
    df = df.fillna(df.mean())
    return df

def preprocess_data(df, target_col='Global_active_power', n_in=60, n_out=1):
    """
    Normalizes data and creates sequences for LSTM.
    n_in: number of previous time steps to use as input
    n_out: number of future time steps to predict
    """
    print("Preprocessing data...")
    
    # Only resample if not already hourly (or based on business logic)
    # Check if the index frequency is already hourly 'H' or 'h'
    if df.index.freq != 'H' and df.index.freq != 'h':
        df_resampled = df.resample('h').mean().fillna(df.mean())
    else:
        df_resampled = df

    values = df_resampled.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # Optimized Sequence Creation using NumPy
    # Create a view of the array with a sliding window
    # Shape: (num_sequences, window_size, num_features)
    num_samples = len(scaled)
    window_size = n_in
    
    # Total sequences possible
    total_sequences = num_samples - window_size - n_out + 1
    
    if total_sequences <= 0:
        return np.array([]), np.array([]), scaler, df_resampled.columns

    # Using strides for efficient windowing
    item_size = scaled.itemsize
    n_features = scaled.shape[1]
    
    # X: (total_sequences, window_size, n_features)
    X = np.lib.stride_tricks.as_strided(
        scaled,
        shape=(total_sequences, window_size, n_features),
        strides=(scaled.strides[0], scaled.strides[0], scaled.strides[1])
    )
    
    # y: (total_sequences, n_out)
    # Target column index
    target_idx = df_resampled.columns.get_loc(target_col)
    
    # The targets start at index window_size
    y_indices = np.arange(window_size, window_size + total_sequences)
    # If n_out > 1, this needs adjustment, but for n_out=1:
    y = scaled[y_indices, target_idx].reshape(-1, n_out)
    
    # Convert X to a copy if needed, though stride_tricks is a view
    return X.copy(), y.copy(), scaler, df_resampled.columns

if __name__ == "__main__":
    # Test loading
    filepath = 'household_power_consumption.txt'
    if os.path.exists(filepath):
        df = load_data(filepath)
        print(df.head())
        X, y, scaler, cols = preprocess_data(df)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
    else:
        print(f"File {filepath} not found.")
