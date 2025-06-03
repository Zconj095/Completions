# Assuming the analysis involves heavy computation not directly shown here
import json
import cupy as cp
import glob
def analyze_data_with_cuda(data):
    # Ensure only numeric values are included for CuPy array creation
    numeric_values = [value for value in data.values() if isinstance(value, (int, float))]
    
    if not numeric_values:
        raise ValueError("Data contains no numeric values for analysis.")
    
    data_gpu = cp.array(numeric_values)
    
    # Perform some analysis computations
    analysis_result_gpu = data_gpu.mean()  # Simplified example
    
    return cp.asnumpy(analysis_result_gpu).item()

def analyze_latest_processed_data():
    list_of_files = glob.glob('processed_data_*.json')  
    latest_file = max(list_of_files, key=lambda x: x.split('_')[2])
    
    with open(latest_file, 'r') as file:
        data = json.load(file)
    
    analysis_result = analyze_data_with_cuda(data)
    
    # Example: Interpret the result and save
    analysis_interpretation = {'health_score': analysis_result}
    analysis_filename = f"analysis_{latest_file.split('_')[2]}"
    with open(analysis_filename, 'w') as file:
        json.dump(analysis_interpretation, file, indent=4)
    print(f"Analysis completed with CUDA and saved to {analysis_filename}.")

if __name__ == "__main__":
    analyze_latest_processed_data()
