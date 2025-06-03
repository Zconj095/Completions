import json
import random
from datetime import datetime

def collect_data():
    # Simulate collecting data from a sensor
    data = {
        'timestamp': datetime.now().isoformat(),
        'heart_rate': random.randint(60, 100),  # Example biometric data
        'skin_temp': random.uniform(36.5, 37.5)  # Example biometric data
    }
    return data

def save_data(data, filename='collected_data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    data = collect_data()
    save_data(data)
    print("Data collected and saved.")
