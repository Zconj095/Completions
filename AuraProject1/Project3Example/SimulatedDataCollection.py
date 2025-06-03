import random
import time
from datetime import datetime

def simulate_aura_data_collection():
    """Simulate collecting aura data from sensors."""
    return {
        'timestamp': datetime.now().isoformat(),
        'heart_rate': random.randint(60, 100),  # Simulated heart rate in BPM
        'skin_temp': random.uniform(36.5, 37.5),  # Simulated skin temperature in Celsius
        'galvanic_skin_response': random.uniform(0.1, 1.0),  # Simulated GSR in microsiemens
    }
