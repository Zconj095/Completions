import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic EM field data
def generate_synthetic_data(length=1000, frequency=5):
    t = np.linspace(0, 1, length)
    data = np.sin(2 * np.pi * frequency * t) + np.random.normal(0, 0.2, length)
    return t, data

time, data = generate_synthetic_data()
plt.plot(time, data)
plt.title("Synthetic EM Field Data")
plt.xlabel("Time")
plt.ylabel("Field Intensity")
plt.show()

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Prepare data for LSTM model
def prepare_data(data, n_features):
    data = data.reshape(len(data), 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(n_features, len(data_scaled)):
        X.append(data_scaled[i-n_features:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

n_features = 50
X, y, scaler = prepare_data(data, n_features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_features, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# Prediction to visualize learning
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10,6))
plt.plot(actual, label='Actual Data')
plt.plot(predicted, label='Predicted Data')
plt.title("Pattern Recognition in EM Field Data")
plt.xlabel("Time Steps")
plt.ylabel("Field Intensity")
plt.legend()
plt.show()

# Forecast future EM field patterns
def forecast(model, last_sequence, n_forecast):
    forecasted = []
    current_sequence = last_sequence
    for _ in range(n_forecast):
        predicted = model.predict(current_sequence[np.newaxis, :, :])
        forecasted.append(predicted[0,0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted
    return scaler.inverse_transform(np.array(forecasted).reshape(-1, 1))

# Using the last sequence from X_test as the input
last_sequence = X_test[-1]
n_forecast = 10  # Number of future time steps to forecast

forecasted_values = forecast(model, last_sequence, n_forecast)
plt.plot(forecasted_values, label='Forecasted EM Field')
plt.title("Forecasted EM Field Patterns")
plt.xlabel("Future Time Steps")
plt.ylabel("Field Intensity")
plt.legend()
plt.show()

def generate_complex_synthetic_data(length=2000, base_frequency=5):
    np.random.seed(42)  # For reproducibility
    time = np.linspace(0, 4, length)
    data = np.sin(2 * np.pi * base_frequency * time) + np.random.normal(0, 0.5, length)  # Base signal with noise
    # Introducing additional patterns
    data += np.sin(2 * np.pi * 2.5 * time)  # Additional frequency component
    data += np.where(time < 2, np.sin(2 * np.pi * 10 * time), 0)  # Sudden change in pattern
    return time, data

time, data = generate_complex_synthetic_data()
plt.plot(time, data)
plt.title("Enhanced Synthetic EM Field Data")
plt.xlabel("Time")
plt.ylabel("Field Intensity")
plt.show()

from keras.preprocessing.sequence import TimeseriesGenerator

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Generate time series sequences
sequence_length = 100
generator = TimeseriesGenerator(data_scaled, data_scaled, length=sequence_length, batch_size=32)

# For simplicity, let's split our data manually into training and testing sets
split_idx = int(0.8 * len(data_scaled))
train_generator = TimeseriesGenerator(data_scaled[:split_idx], data_scaled[:split_idx],
                                      length=sequence_length, batch_size=32)
test_generator = TimeseriesGenerator(data_scaled[split_idx:], data_scaled[split_idx:],
                                     length=sequence_length, batch_size=32)

from keras.layers import Dropout
from keras.regularizers import L1L2

model = Sequential([
    LSTM(100, activation='relu', input_shape=(sequence_length, 1), return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(train_generator, epochs=50, validation_data=test_generator)

