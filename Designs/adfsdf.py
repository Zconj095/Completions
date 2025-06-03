import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Enhanced Emotion Class
class EnhancedEmotion:
    def __init__(self, name, intensity, description):
        self.name = name
        self.intensity = intensity
        self.description = description
        self.timestamp = pd.Timestamp.now()
    
    def describe(self):
        return f"Emotion: {self.name}, Intensity: {self.intensity:.2f}, Description: {self.description}"

# Emotion prediction using sentiment analysis
def predict_emotional_state(user_input):
    """Predicts emotional state using NLTK sentiment analysis"""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(user_input)
    
    compound = sentiment_scores['compound']
    
    if compound >= 0.6:
        return EnhancedEmotion("Joy", compound, "Very positive sentiment")
    elif compound >= 0.2:
        return EnhancedEmotion("Happiness", compound, "Positive sentiment")
    elif compound <= -0.6:
        return EnhancedEmotion("Sadness", abs(compound), "Very negative sentiment")
    elif compound <= -0.2:
        return EnhancedEmotion("Concern", abs(compound), "Negative sentiment")
    else:
        return EnhancedEmotion("Neutral", abs(compound), "Neutral sentiment")

# Simple LSTM-like model using basic ML
class SimpleLSTMModel:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.emotion_history = []
    
    def extract_features(self, data):
        """Extract features from emotional data"""
        if isinstance(data, str):
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(data)
            return np.array([[scores['compound'], scores['pos'], scores['neg'], scores['neu']]])
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            # Handle list of strings
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(data[0])
            return np.array([[scores['compound'], scores['pos'], scores['neg'], scores['neu']]])
        else:
            # Ensure data has the right shape for 4 features
            data_array = np.array(data)
            if data_array.size == 4:
                return data_array.reshape(-1, 4)
            elif data_array.size > 4:
                return data_array.reshape(-1, 4)
            else:
                # Pad with zeros if not enough features
                padded = np.zeros(4)
                padded[:data_array.size] = data_array.flatten()
                return padded.reshape(-1, 4)
    
    def fit(self, data, emotions, epochs=100, batch_size=32):
        """Train the model"""
        features = []
        targets = []
        
        for i, text in enumerate(data):
            feature = self.extract_features(text)
            features.append(feature[0])
            targets.append(emotions[i])
        
        self.features = np.array(features)
        self.targets = np.array(targets)
        self.scaler.fit(self.features)
        self.is_trained = True
        print(f"Model trained on {len(data)} samples")
    
    def predict(self, data):
        """Predict emotional state"""
        if not self.is_trained:
            # Fallback to sentiment analysis
            return predict_emotional_state(data[0] if isinstance(data, list) else data)
        
        features = self.extract_features(data)
        scaled_features = self.scaler.transform(features)
        
        # Simple prediction logic (replace with actual LSTM in production)
        prediction_score = np.mean(scaled_features[0])
        
        if prediction_score > 0.3:
            return EnhancedEmotion("Positive", prediction_score, "Predicted positive emotion")
        elif prediction_score < -0.3:
            return EnhancedEmotion("Negative", abs(prediction_score), "Predicted negative emotion")
        else:
            return EnhancedEmotion("Neutral", abs(prediction_score), "Predicted neutral emotion")

# Data processing functions
def clean_data(data):
    """Clean and preprocess data"""
    if isinstance(data, list):
        return [str(item).strip() for item in data if item and str(item).strip()]
    return data

def preprocess_data(data):
    """Preprocess data for model input"""
    cleaned_data = clean_data(data)
    return cleaned_data

def preprocess_user_data(user_data):
    """Preprocess user input data"""
    return preprocess_data(user_data)

# Utility functions
def interpret_emotion(predicted_emotion):
    """Interpret emotion and provide feedback"""
    emotion_name = predicted_emotion.name.lower()
    
    feedback_map = {
        'joy': "You seem very happy! Keep up the positive energy!",
        'happiness': "Great to see you're feeling positive!",
        'sadness': "It seems you're feeling down. Consider talking to someone or doing something you enjoy.",
        'concern': "You might be experiencing some negative feelings. Take care of yourself.",
        'neutral': "You seem to be in a balanced emotional state.",
        'positive': "Your emotional trend looks positive!",
        'negative': "Your emotional trend suggests some challenges. Consider self-care activities."
    }
    
    return feedback_map.get(emotion_name, "Your emotional state has been noted.")

def interpret_emotion_prediction(prediction):
    """Interpret model prediction"""
    return interpret_emotion(prediction)

# Data collection functions
def collect_user_data():
    """Collect user input data"""
    try:
        user_input = input("How are you feeling? Describe your current state: ")
        return user_input if user_input.strip() else "I'm feeling okay"
    except (EOFError, KeyboardInterrupt):
        return "I'm feeling okay"

def collect_latest_user_data():
    """Collect latest user data"""
    return collect_user_data()

def collect_user_feedback():
    """Collect user feedback"""
    try:
        feedback = input("Was this prediction helpful? (yes/no): ")
        return feedback.lower().startswith('y')
    except (EOFError, KeyboardInterrupt):
        return True

# Display functions
def display_prediction_to_user(prediction):
    """Display prediction to user"""
    print(f"\n--- Emotional Analysis ---")
    print(prediction.describe())
    print(f"Feedback: {interpret_emotion(prediction)}")
    print("-" * 30)

def present_feedback_to_user(feedback):
    """Present feedback to user"""
    print(f"\nFeedback: {feedback}")

# Main functions
def emotional_analysis_process():
    """Main emotional analysis process"""
    user_input = collect_user_data()
    predicted_emotion = predict_emotional_state(user_input)
    display_prediction_to_user(predicted_emotion)

def emotional_state_forecast():
    """Forecast emotional state using ML model"""
    global lstm_model
    
    latest_user_data = collect_latest_user_data()
    predicted_emotion = lstm_model.predict([latest_user_data])
    feedback = interpret_emotion(predicted_emotion)
    present_feedback_to_user(feedback)

def predict_user_emotion(user_data):
    """Predict user emotion using trained model"""
    global lstm_model
    processed_data = preprocess_user_data([user_data])
    predicted_emotion = lstm_model.predict(processed_data)
    return predicted_emotion

def main_interaction_loop():
    """Main interaction loop"""
    global lstm_model
    
    print("Welcome to the Emotional Analysis System!")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_data = collect_user_data()
            
            if user_data.lower() in ['quit', 'exit', 'stop']:
                print("Thank you for using the Emotional Analysis System!")
                break
            
            # Use simple sentiment analysis or trained model
            if lstm_model.is_trained:
                predicted_emotion = predict_user_emotion(user_data)
            else:
                predicted_emotion = predict_emotional_state(user_data)
            
            display_prediction_to_user(predicted_emotion)
            
            # Optional feedback collection
            helpful = collect_user_feedback()
            if helpful:
                print("Thank you for your feedback!")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

# Initialize the model
lstm_model = SimpleLSTMModel()

# Example training data (replace with real data)
sample_texts = [
    "I'm feeling great today!",
    "This is terrible, I hate everything",
    "Just another ordinary day",
    "I'm so excited about this project!",
    "I'm feeling a bit down today"
]

sample_emotions = ["happy", "sad", "neutral", "excited", "sad"]

# Train the model with sample data
lstm_model.fit(sample_texts, sample_emotions)

# Run the main program
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Single analysis")
    print("2. Continuous interaction")
    print("3. Forecast mode")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            emotional_analysis_process()
        elif choice == "2":
            main_interaction_loop()
        elif choice == "3":
            emotional_state_forecast()
        else:
            print("Invalid choice, running single analysis...")
            emotional_analysis_process()
    except (EOFError, KeyboardInterrupt):
        emotional_analysis_process()
