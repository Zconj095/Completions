import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class EnhancedEmotion:
    def __init__(self, name, intensity, impact):
        self.name = name
        self.intensity = intensity
        self.impact = impact
        self.timestamp = None
    
    def describe(self):
        return f"Emotion: {self.name}, Intensity: {self.intensity}/10, Impact: {self.impact}"

class EnhancedMood:
    def __init__(self, name, duration, effect):
        self.name = name
        self.duration = duration
        self.effect = effect
    
    def describe(self):
        return f"Mood: {self.name}, Duration: {self.duration}, Effect: {self.effect}"

class EnhancedFeeling:
    def __init__(self, name, trigger):
        self.name = name
        self.trigger = trigger
    
    def describe(self):
        return f"Feeling: {self.name}, Triggered by: {self.trigger}"

class EnhancedBelief:
    def __init__(self, name, category, influence):
        self.name = name
        self.category = category
        self.influence = influence
    
    def describe(self):
        return f"Belief: {self.name}, Category: {self.category}, Influence: {self.influence}"

def analyze_user_state(emotion, mood):
    """Analyzes the combination of emotion and mood"""
    analysis = f"Analysis: Your {emotion.name} emotion (intensity {emotion.intensity}) "
    analysis += f"combined with {mood.name} mood suggests "
    
    if emotion.intensity > 7:
        analysis += "a strong emotional state requiring attention."
    elif emotion.intensity > 4:
        analysis += "a moderate emotional state that's manageable."
    else:
        analysis += "a mild emotional state that's relatively stable."
    
    return analysis

def complex_analysis(emotion, mood, feeling, belief):
    """Performs complex analysis of all emotional components"""
    analysis = f"Comprehensive Analysis:\n"
    analysis += f"- {emotion.describe()}\n"
    analysis += f"- {mood.describe()}\n"
    analysis += f"- {feeling.describe()}\n"
    analysis += f"- {belief.describe()}\n"
    
    # Simple correlation analysis
    if emotion.intensity > 6 and "positive" in mood.effect.lower():
        analysis += "\nRecommendation: Channel this positive energy into productive activities."
    elif emotion.intensity > 6 and "negative" in mood.effect.lower():
        analysis += "\nRecommendation: Consider stress management techniques."
    else:
        analysis += "\nRecommendation: Maintain current emotional balance."
    
    return analysis

def get_user_emotional_state():
    """Enhanced user input collection with validation"""
    print("=== Emotional State Assessment ===")
    
    emotion_name = input("Enter your current primary emotion (e.g., joy, anger, fear): ").strip()
    while not emotion_name:
        emotion_name = input("Please enter a valid emotion: ").strip()
    
    try:
        intensity = int(input("Rate intensity (1-10): "))
        intensity = max(1, min(10, intensity))  # Clamp between 1-10
    except ValueError:
        intensity = 5
        print("Invalid input, using default intensity of 5")
    
    mood_name = input("Enter your current mood (e.g., optimistic, gloomy, calm): ").strip()
    
    user_emotion = EnhancedEmotion(emotion_name, intensity, "User-defined impact")
    user_mood = EnhancedMood(mood_name, "Current session", "User-defined effect")
    
    return user_emotion, user_mood

def predict_emotional_state(user_input):
    """Predicts emotional state using sentiment analysis"""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(user_input)
    
    compound = sentiment_scores['compound']
    intensity = abs(compound) * 10  # Scale to 1-10
    
    if compound > 0.5:
        return EnhancedEmotion("Happiness", intensity, "Positive sentiment detected")
    elif compound < -0.5:
        return EnhancedEmotion("Sadness", intensity, "Negative sentiment detected")
    elif compound > 0.1:
        return EnhancedEmotion("Contentment", intensity, "Mild positive sentiment")
    elif compound < -0.1:
        return EnhancedEmotion("Concern", intensity, "Mild negative sentiment")
    else:
        return EnhancedEmotion("Neutral", 0, "Neutral sentiment")

def emotional_analysis_process():
    """Main emotional analysis workflow"""
    print("\n=== Starting Emotional Analysis ===")
    
    # Method 1: User input
    user_emotion, user_mood = get_user_emotional_state()
    
    # Method 2: Text analysis
    text_input = input("\nDescribe how you're feeling in your own words: ")
    predicted_emotion = predict_emotional_state(text_input)
    
    # Create additional components
    user_feeling = EnhancedFeeling("Contentment", "Achieving goals")
    user_belief = EnhancedBelief("Growth mindset", "Personal development", "Drives continuous learning")
    
    print("\n=== Analysis Results ===")
    print("User-reported state:")
    print(user_emotion.describe())
    print(user_mood.describe())
    
    print(f"\nText-based prediction:")
    print(predicted_emotion.describe())
    
    # Comprehensive analysis
    analysis_result = complex_analysis(user_emotion, user_mood, user_feeling, user_belief)
    print(f"\n{analysis_result}")

# Main execution
if __name__ == "__main__":
    emotional_analysis_process()
