import os

try:
    if os.environ.get('RENDER'):
        raise ImportError("Bypassing TensorFlow on Render to stay under 512MB memory limit")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

def build_lstm_model(input_shape):
    """
    Builds a multi-layer LSTM model as specified in the abstract.
    input_shape: (time_steps, features)
    """
    if not HAS_TENSORFLOW:
        return None
    model = Sequential()
    
    # Layer 1
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Layer 2
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
