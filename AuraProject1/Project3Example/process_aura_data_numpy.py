import numpy as np
from SimulatedDataCollection import simulate_aura_data_collection

def process_aura_data_numpy(data_samples):
    """Process a batch of collected aura data using NumPy for statistical analysis."""
    
    # Define the dtype for the structured array
    dtype = [('timestamp', 'datetime64[s]'), ('heart_rate', 'f8'), ('skin_temp', 'f8'), ('galvanic_skin_response', 'f8')]
    
    # Preallocate numpy structured array
    data_array = np.empty(len(data_samples), dtype=dtype)
    
    # Manually fill the structured array
    for i, sample in enumerate(data_samples):
        # Here, ensuring timestamp is correctly converted to numpy.datetime64
        data_array[i] = (np.datetime64(sample['timestamp']), sample['heart_rate'], sample['skin_temp'], sample['galvanic_skin_response'])
    
    # Example processing: Calculate mean heart rate and temperature
    mean_heart_rate = np.mean(data_array['heart_rate'])
    mean_skin_temp = np.mean(data_array['skin_temp'])
    
    print(f"Mean Heart Rate: {mean_heart_rate}, Mean Skin Temperature: {mean_skin_temp}")
    return mean_heart_rate, mean_skin_temp

# Simulate batch data processing
data_samples = [simulate_aura_data_collection() for _ in range(10)]
process_aura_data_numpy(data_samples)
