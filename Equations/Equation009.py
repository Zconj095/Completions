import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class EMFieldClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_data(self, X, y):
        """Prepare and scale the data properly"""
        # Split first, then scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Fit scaler only on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Find optimal SVM hyperparameters"""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train(self, X, y):
        """Train the model with hyperparameter optimization"""
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Optimize hyperparameters
        self.model = self.optimize_hyperparameters(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Final training
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluation
        self.evaluate(X_test, y_test)
        
        return X_test, y_test
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.model.predict(X_test)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
    def classify_em_field(self, features, return_probability=False):
        """Classify new EM field data with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        
        if return_probability and hasattr(self.model, 'predict_proba'):
            # For probability estimates, use SVC with probability=True
            probabilities = self.model.predict_proba(features_scaled)[0]
            return prediction, probabilities
        
        return prediction
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and scaler"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        print(f"Model loaded from {filepath}")

# Enhanced dataset with more realistic EM field features
def generate_sample_data():
    """Generate more comprehensive EM field sample data"""
    np.random.seed(42)
    
    # Features: [frequency, amplitude, phase, polarization]
    # Class 1: Low frequency sources
    class1 = np.random.normal([1, 2, 0, 0], [0.5, 0.3, 0.2, 0.1], (50, 4))
    
    # Class 2: Medium frequency sources  
    class2 = np.random.normal([5, 4, 1, 0.5], [0.8, 0.5, 0.3, 0.2], (50, 4))
    
    # Class 3: High frequency sources
    class3 = np.random.normal([10, 6, 2, 1], [1.0, 0.7, 0.4, 0.3], (50, 4))
    
    X = np.vstack([class1, class2, class3])
    y = np.hstack([np.ones(50), np.ones(50)*2, np.ones(50)*3])
    
    return X, y.astype(int)

# Main execution
if __name__ == "__main__":
    # Generate sample data
    X, y = generate_sample_data()
    print(f"Dataset shape: {X.shape}, Classes: {np.unique(y)}")
    
    # Create and train classifier
    classifier = EMFieldClassifier()
    X_test, y_test = classifier.train(X, y)
    
    # Example predictions
    test_features = [
        [2, 3, 0.5, 0.2],  # Should be class 1
        [6, 5, 1.2, 0.6],  # Should be class 2
        [11, 7, 2.1, 1.1]  # Should be class 3
    ]
    
    print("\nPredictions on new data:")
    for i, features in enumerate(test_features):
        prediction = classifier.classify_em_field(features)
        print(f"Features {features} -> Predicted class: {prediction}")
    
    # Save model for future use
    classifier.save_model("em_field_classifier.pkl")