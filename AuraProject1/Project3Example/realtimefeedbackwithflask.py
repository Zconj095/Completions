from flask import Flask, jsonify
import numpy as np
import datetime

app = Flask(__name__)

def simulate_aura_data_collection():
    """
    Simulate fetching the latest processed aura data.
    Returns a dictionary mimicking the structure of actual processed data.
    """
    # Simulate data with current timestamp, random heart rate, skin temp, and GSR
    simulated_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'heart_rate': np.random.randint(60, 100),  # Random heart rate between 60 and 100
        'skin_temp': np.random.uniform(36.5, 37.5),  # Random skin temperature between 36.5 and 37.5 degrees Celsius
        'galvanic_skin_response': np.random.uniform(0.1, 1.0),  # Random GSR between 0.1 and 1.0 microsiemens
    }
    return simulated_data

def generate_feedback(data):
    """
    Generate feedback based on processed data.
    """
    feedback_message = "Your aura data is within normal ranges."
    if data['heart_rate'] > 80:
        feedback_message = "Your heart rate is elevated. Consider taking a moment to relax."
    return {"feedback": feedback_message}

@app.route('/api/feedback')
def get_real_time_feedback():
    """
    Endpoint to get real-time feedback based on the latest processed aura data.
    """
    processed_data = simulate_aura_data_collection()  # Fetch simulated processed data
    feedback = generate_feedback(processed_data)  # Generate feedback based on processed data
    return jsonify(feedback)

if __name__ == '__main__':
    app.run(debug=True)