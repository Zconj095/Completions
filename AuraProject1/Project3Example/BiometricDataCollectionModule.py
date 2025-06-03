import json
from datetime import datetime
import random

def collect_biometric_data():
    """Simulate biometric data collection."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'heart_rate': random.randint(60, 100),  # Simulated heart rate
        'skin_temp': random.uniform(36.5, 37.5)  # Simulated skin temperature
    }
    filename = f"data_{data['timestamp'].replace(':', '-')}.json"
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Biometric data collected and saved to {filename}.")

if __name__ == "__main__":
    collect_biometric_data()
