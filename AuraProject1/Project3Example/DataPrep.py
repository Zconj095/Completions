import numpy as np
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Assuming 'data' is your time series data loaded as a numpy array
seq_length = 5  # Example sequence length
x, y = create_sequences(data, seq_length)
