import json
import glob

def analyze_processed_data():
    """Analyze the latest processed data file."""
    list_of_files = glob.glob('processed_*.json')
    latest_file = max(list_of_files, key=lambda x: x.split('_')[2])
    
    with open(latest_file, 'r') as file:
        data = json.load(file)
    
    # Simulate analysis: Determine health status based on normalized heart rate
    health_status = "Healthy" if data['normalized_heart_rate'] > 0.5 else "Check-up Recommended"
    
    # Saving analysis result
    analysis_result = {'health_status': health_status}
    analysis_filename = f"analysis_result_{latest_file.split('_')[2]}"
    with open(analysis_filename, 'w') as file:
        json.dump(analysis_result, file, indent=4)
    print(f"Analysis completed and saved to {analysis_filename}.")

if __name__ == "__main__":
    analyze_processed_data()
