import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Assuming 'data' is your loaded time series dataset

# Generate synthetic time series data
# For example, 100 data points representing some measure over time
data = np.sin(np.linspace(0, 20, 100))  # Sine wave as example data

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# Create sequences
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

seq_length = 5  # Example sequence length
sequences = create_inout_sequences(data_normalized, seq_length)
