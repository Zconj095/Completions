import json
from DataProcessingModule import *
def load_processed_data(filename='processed_data.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def analyze_data(data):
    # Simulate data analysis (insert your analysis logic here)
    analysis_result = {'health_status': 'normal' if data['heart_rate'] > 0.5 else 'checkup_required'}
    return analysis_result

def save_analysis_results(results, filename='analysis_results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    data = load_processed_data()
    results = analyze_data(data)
    save_analysis_results(results)
    print("Analysis completed and results saved.")
