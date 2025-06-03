import json
import cupy as cp
import glob

def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def process_data_with_cuda(data):
    # Example: Convert heart rate and skin temp to CuPy arrays for GPU processing
    heart_rate_array = cp.array(data['heart_rate'])
    skin_temp_array = cp.array(data['skin_temp'])
    
    # Simulate processing: Normalize heart rate and temperature
    normalized_heart_rate = (heart_rate_array - 60) / (100 - 60)
    normalized_skin_temp = (skin_temp_array - 36.5) / (37.5 - 36.5)
    
    # Convert back to Python scalars for JSON serialization
    data['normalized_heart_rate'] = cp.asnumpy(normalized_heart_rate).item()
    data['normalized_skin_temp'] = cp.asnumpy(normalized_skin_temp).item()
    return data

def process_latest_data():
    list_of_files = glob.glob('data_*.json')  
    latest_file = max(list_of_files, key=lambda x: x.split('_')[1])
    data = load_data(latest_file)
    
    processed_data = process_data_with_cuda(data)
    
    processed_filename = f"processed_{latest_file}"
    with open(processed_filename, 'w') as file:
        json.dump(processed_data, file, indent=4)
    print(f"Processed data with CUDA and saved to {processed_filename}.")

if __name__ == "__main__":
    process_latest_data()
