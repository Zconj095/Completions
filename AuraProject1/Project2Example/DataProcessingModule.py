import json

def load_data(filename='collected_data.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    # Simulate data preprocessing (this is where your logic would go)
    data['heart_rate'] = (data['heart_rate'] - 60) / (100 - 60)  # Example normalization
    return data

def save_processed_data(data, filename='processed_data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    raw_data = load_data()
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data)
    print("Data processed and saved.")
