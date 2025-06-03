import numpy as np
import matplotlib.pyplot as plt
import random

def generate_enhanced_synthetic_data(length=2000, frequencies=[5, 10], noise_level=0.2, random_seed=42):
    np.random.seed(random_seed)
    t = np.linspace(0, 4, length)
    data = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    data += np.random.normal(0, noise_level, length)
    return t, data

t, data = generate_enhanced_synthetic_data()
plt.plot(t, data)
plt.title('Enhanced Synthetic EM Field Data')
plt.xlabel('Time')
plt.ylabel('Field Intensity')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Step 1: Generate synthetic sine wave data with noise
def generate_data(seq_length=200):
    np.random.seed(0)  # For reproducibility
    time_step = np.linspace(0, 20, seq_length)
    data = np.sin(time_step) + np.random.normal(scale=0.5, size=seq_length)
    return data

data = generate_data()

# Step 2: Prepare the data for LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 20
X, y = prepare_data(data, n_steps)
# Reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Step 3: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Step 4: Define the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model (use a small number of epochs for demonstration purposes)
model.fit(X_scaled, y_scaled, epochs=5, verbose=0)

# Animation
fig, ax = plt.subplots()
ax.set_title('Real-time LSTM Predictions')
ax.set_xlim(0, len(data))
ax.set_ylim(np.min(data), np.max(data))
line, = ax.plot([], [], 'g-', linewidth=2, label='LSTM Predictions')
line_true, = ax.plot(data, 'r-', alpha=0.5, label='True Data')
ax.legend()

def animate(i):
    if i > n_steps:
        X_test = data[i-n_steps:i]
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(1, n_steps, n_features)
        y_pred = model.predict(X_test_scaled)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        line.set_data(range(i-n_steps, i+1), np.append(X_test, y_pred_rescaled[0]))
    return line,

ani = FuncAnimation(fig, animate, frames=len(data), blit=True, interval=100)

plt.show()

def generate_complex_data(seq_length=1000):
    np.random.seed(0)  # For reproducibility
    time_step = np.linspace(0, 50, seq_length)
    # Creating a more complex pattern
    data = np.sin(time_step) * np.cos(time_step / 2) + np.random.normal(scale=0.5, size=seq_length)
    return data

data = generate_complex_data()

def sliding_window(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 30  # Increased window size for capturing more complex patterns
X, y = sliding_window(data, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Ensure TensorFlow is using GPU (if available)
tf.config.list_physical_devices('GPU')

# Generate synthetic time series data
def generate_data(length=1200, cycle_length=50):
    np.random.seed(42)
    time = np.linspace(0, cycle_length, length)
    data = np.sin(time) + np.random.normal(scale=0.5, size=length)
    return data

data = generate_data()

# Prepare the dataset
def prepare_dataset(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps_in, n_steps_out = 30, 10
X, y = prepare_dataset(data, n_steps_in, n_steps_out)
X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects [samples, timesteps, features]

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = np.array([scaler.fit_transform(x) for x in X])
y_scaled = np.array([scaler.fit_transform(y[i].reshape(-1, 1)).reshape(-1) for i in range(len(y))])

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps_in, 1)),
    Dense(n_steps_out)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_scaled, y_scaled, epochs=20, verbose=1)

# Visualization with Matplotlib Animation
fig, ax = plt.subplots()
ax.set_title("Time Series Forecasting")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
line, = ax.plot([], [], 'r-', linewidth=2, label='Predicted')
line_true, = ax.plot(data, 'g-', alpha=0.5, label='True Data')
ax.legend()

def init():
    line.set_data([], [])
    return line, line_true,

def animate(i):
    if i >= n_steps_in and i + n_steps_out <= len(data):
        X_test = scaler.transform(data[i-n_steps_in:i].reshape(-1, 1)).reshape(1, n_steps_in, 1)
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred).flatten()
        line.set_data(range(i, i + n_steps_out), y_pred_rescaled)
    return line, line_true,

ani = FuncAnimation(fig, animate, frames=np.arange(n_steps_in, len(data)-n_steps_out, n_steps_out), init_func=init, blit=True, interval=100)

plt.show()
