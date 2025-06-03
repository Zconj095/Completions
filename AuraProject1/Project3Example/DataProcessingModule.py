import json
import glob

def process_latest_biometric_data():
    """Find the most recent data file and process it."""
    list_of_files = glob.glob('*.json')  # Assuming data is collected in the current directory
    latest_file = max(list_of_files, key=lambda x: x.split('_')[1])
    
    with open(latest_file, 'r') as file:
        data = json.load(file)
    
    # Example processing: Normalizing heart rate
    data['normalized_heart_rate'] = (data['heart_rate'] - 60) / 40
    
    # Save processed data back to a new file
    processed_filename = f"processed_{latest_file}"
    with open(processed_filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Processed data saved to {processed_filename}.")

if __name__ == "__main__":
    process_latest_biometric_data()
