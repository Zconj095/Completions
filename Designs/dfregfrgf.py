import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_sequences(data, sequence_length=60):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def preprocess_data(data, target_column, sequence_length=60):
    """Enhanced data preprocessing with proper sequence creation"""
    # Extract target column
    target_data = data[target_column].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target_data)
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    return X, y, scaler

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Enhanced LSTM model with dropout for regularization"""
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units),
        Dropout(dropout_rate),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def train_and_evaluate_model(data_file='your_data_file.csv', target_column='price'):
    """Complete training and evaluation pipeline"""
    try:
        # Load data
        data = pd.read_csv(data_file)
        print(f"Data shape: {data.shape}")
        
        # Preprocess data
        X, y, scaler = preprocess_data(data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Build model
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Inverse transform predictions
        train_pred = scaler.inverse_transform(train_pred)
        test_pred = scaler.inverse_transform(test_pred)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        train_mae = mean_absolute_error(y_train_actual, train_pred)
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        
        print(f"\nTraining RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        return model, history, scaler, (X_test, y_test_actual, test_pred)
        
    except FileNotFoundError:
        print(f"File {data_file} not found. Please provide a valid CSV file.")
        return None, None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

# Run the enhanced pipeline
if __name__ == "__main__":
    model, history, scaler, test_results = train_and_evaluate_model()
