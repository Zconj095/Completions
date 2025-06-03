import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Optional imports with error handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Embedding
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    print("CuPy not available for CUDA acceleration. Install with: pip install cupy")
    CUPY_AVAILABLE = False

class GenreClassificationPipeline:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.models = {}
        
    def create_sample_dataset(self, save_path='genre_dataset.csv'):
        """Create and save a sample dataset"""
        data = {
            'ID': np.arange(1, 101),
            'Date': pd.date_range(start='2020-01-01', periods=100),
            'Content': [
                "In a galaxy far, far away, a small rebellion fights against tyranny...",
                "She opened the ancient spellbook and magic energy flowed out...",
                "The detective examined the crime scene looking for hidden clues...",
                "Advanced AI robots developed consciousness and started questioning...",
                "The young wizard's apprentice accidentally cast a powerful spell...",
                "Love bloomed unexpectedly between two unlikely souls...",
                "The spaceship encountered an alien civilization on a distant planet...",
                "Dark forces gathered in the enchanted forest threatening the kingdom...",
                "A murder occurred in the locked library room with no witnesses...",
                "Time travel technology created paradoxes across multiple timelines..."
            ] * 10,  # Repeat to get 100 samples
            'Genre': [
                "Science Fiction", "Fantasy", "Mystery", "Science Fiction", "Fantasy",
                "Romance", "Science Fiction", "Fantasy", "Mystery", "Science Fiction"
            ] * 10
        }
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Dataset created and saved as {save_path}")
        return df
    
    def preprocess_text_data(self, data):
        """Preprocess text data for machine learning"""
        # Convert text to TF-IDF features
        tfidf_features = self.tfidf.fit_transform(data['Content']).toarray()
        
        # Encode genre labels
        genre_labels = self.label_encoder.fit_transform(data['Genre'])
        
        return tfidf_features, genre_labels
    
    def train_decision_tree(self, X, y):
        """Train Decision Tree Classifier"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        self.models['decision_tree'] = clf
        print(f"Decision Tree Accuracy: {accuracy:.4f}")
        return clf, accuracy
    
    def train_random_forest(self, X, y):
        """Train Random Forest Classifier"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        self.models['random_forest'] = rf
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        return rf, accuracy
    
    def train_xgboost_model(self, X, y):
        """Train XGBoost model if available"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available, skipping...")
            return None, 0
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y))
        }
        
        model_xgb = xgb.train(params, dtrain, num_boost_round=100)
        predictions = model_xgb.predict(dtest)
        accuracy = accuracy_score(y_test, predictions)
        
        self.models['xgboost'] = model_xgb
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        return model_xgb, accuracy
    
    def train_neural_network(self, data):
        """Train Neural Network model if TensorFlow is available"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available, skipping neural network...")
            return None, 0
        
        # Tokenize text
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(data['Content'])
        sequences = tokenizer.texts_to_sequences(data['Content'])
        X = pad_sequences(sequences, maxlen=100)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(data['Genre'])
        y = to_categorical(y_encoded)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        model = Sequential([
            Embedding(5000, 64, input_length=100),
            LSTM(64, dropout=0.5, recurrent_dropout=0.5),
            Dense(32, activation='relu'),
            Dense(len(np.unique(self.label_encoder.classes_)), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                           validation_data=(X_test, y_test), verbose=0)
        
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        self.models['neural_network'] = model
        self.tokenizer = tokenizer
        print(f"Neural Network Accuracy: {accuracy:.4f}")
        return model, accuracy
    
    def predict_genre(self, text, model_type='decision_tree'):
        """Predict genre for new text"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        if model_type == 'neural_network':
            if not hasattr(self, 'tokenizer'):
                raise ValueError("Tokenizer not available for neural network prediction")
            
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=100)
            prediction = self.models[model_type].predict(padded)
            predicted_class = np.argmax(prediction)
        else:
            # For traditional ML models
            text_features = self.tfidf.transform([text]).toarray()
            
            if model_type == 'xgboost':
                dtest = xgb.DMatrix(text_features)
                prediction = self.models[model_type].predict(dtest)
                predicted_class = int(prediction[0])
            else:
                predicted_class = self.models[model_type].predict(text_features)[0]
        
        predicted_genre = self.label_encoder.inverse_transform([predicted_class])[0]
        return predicted_genre
    
    def cuda_preprocess_data(self, data):
        """CUDA-accelerated preprocessing if CuPy is available"""
        if not CUPY_AVAILABLE:
            return data
        
        if not isinstance(data, cp.ndarray):
            gpu_data = cp.asarray(data)
        else:
            gpu_data = data
        
        mean = gpu_data.mean(axis=0)
        std = gpu_data.std(axis=0)
        normalized_data = (gpu_data - mean) / std
        
        return cp.asnumpy(normalized_data)
    
    def time_series_forecast(self, data):
        """Time series forecasting with Prophet if available"""
        if not PROPHET_AVAILABLE:
            print("Prophet not available for time series forecasting")
            return None
        
        # Prepare data for Prophet
        genre_counts = data.groupby(['Date', 'Genre']).size().reset_index(name='count')
        
        forecasts = {}
        for genre in genre_counts['Genre'].unique():
            genre_data = genre_counts[genre_counts['Genre'] == genre]
            df_prophet = pd.DataFrame({
                'ds': genre_data['Date'],
                'y': genre_data['count']
            })
            
            if len(df_prophet) > 1:  # Need at least 2 data points
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                forecasts[genre] = forecast
        
        return forecasts

def main():
    # Initialize pipeline
    pipeline = GenreClassificationPipeline()
    
    # Create or load dataset
    try:
        data = pd.read_csv('genre_dataset.csv')
        print("Loaded existing dataset")
    except FileNotFoundError:
        print("Creating new dataset...")
        data = pipeline.create_sample_dataset()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y = pipeline.preprocess_text_data(data)
    
    # Train models
    print("\nTraining models...")
    dt_model, dt_acc = pipeline.train_decision_tree(X, y)
    rf_model, rf_acc = pipeline.train_random_forest(X, y)
    xgb_model, xgb_acc = pipeline.train_xgboost_model(X, y)
    nn_model, nn_acc = pipeline.train_neural_network(data)
    
    # Time series forecasting
    print("\nPerforming time series forecasting...")
    forecasts = pipeline.time_series_forecast(data)
    
    # Example prediction
    print("\nExample predictions:")
    test_text = "The spaceship encountered aliens on a distant planet with advanced technology"
    
    for model_name in ['decision_tree', 'random_forest']:
        if model_name in pipeline.models:
            prediction = pipeline.predict_genre(test_text, model_name)
            print(f"{model_name.replace('_', ' ').title()}: {prediction}")
    
    if XGBOOST_AVAILABLE and 'xgboost' in pipeline.models:
        prediction = pipeline.predict_genre(test_text, 'xgboost')
        print(f"XGBoost: {prediction}")
    
    if TENSORFLOW_AVAILABLE and 'neural_network' in pipeline.models:
        prediction = pipeline.predict_genre(test_text, 'neural_network')
        print(f"Neural Network: {prediction}")

if __name__ == "__main__":
    main()
