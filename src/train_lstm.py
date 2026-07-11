"""
LSTM Training Module for Financial Time-Series Forecasting
Author: Ozgur Saracik
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def build_lstm_model(input_shape):
    """
    Builds and compiles a 4-layer LSTM architecture.
    """
    model = Sequential()
    
    # First LSTM Layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM Layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense Layers for prediction
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1)) # Final output layer for price prediction
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Trains the LSTM model with specified hyperparameters.
    """
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    return history

if __name__ == "__main__":
    # Placeholder for input dimensions (e.g., 60 timesteps, 1 feature)
    sample_input_shape = (60, 1)
    lstm_model = build_lstm_model(sample_input_shape)
    lstm_model.summary()
