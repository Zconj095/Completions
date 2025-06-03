import numpy as np
def preprocess_data(data):
    # Example: Normalize data using Min-Max scaling
    min_val = np.min(data)
    max_val = np.max(data)
    preprocessed_data = (data - min_val) / (max_val - min_val)
    return preprocessed_data
