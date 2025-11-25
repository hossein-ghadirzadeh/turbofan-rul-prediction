import tensorflow as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Tuple

def build_lstm_model(input_shape: Tuple[int, int], learning_rate: float = 0.001):
    """
    Builds and compiles a Stacked LSTM model for RUL prediction.
    
    Args:
        input_shape: (time_steps, features) -> (30, 14)
        learning_rate: Optimizer learning rate
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Layer 1: LSTM with return_sequences=True (to feed into the next LSTM layer)
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Layer 2: LSTM without return_sequences (Last hidden state only)
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer 3: Dense layers for regression
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1)) 
    
    optimizer = Adam(learning_rate=learning_rate)
    
    # Loss function = Mean Squared Error
    # MAE (Mean Absolute Error) 
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model