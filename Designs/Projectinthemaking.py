"""
Enhanced Aura Analysis and Prediction Model
Combines electromagnetic, psychophysiological, neurocognitive, and symbolic models
for comprehensive aura analysis with machine learning predictions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

class AuraAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.feature_selector = SelectKBest(f_regression, k=10)
        
    def calculate_ea(self, C, P, F):
        """Calculate electromagnetic aura intensity"""
        return C * 0.4 + P * 0.3 + F * 0.3
    
    def calculate_as(self, E, S, T):
        """Calculate aural sensations in psychophysiological model"""
        return E * 0.4 + S * 0.3 + T * 0.3
    
    def calculate_va(self, I, C, N):
        """Calculate visualized aura in neurocognitive model"""
        return I * 0.4 + C * 0.3 + N * 0.3
    
    def calculate_sa(self, M, P, E):
        """Calculate symbolic meaning of the aura"""
        return M * 0.4 + P * 0.3 + E * 0.3
    
    def generate_sample_data(self, n_samples=1000):
        """Generate enhanced sample data with more realistic distributions"""
        np.random.seed(self.random_state)
        
        data = {
            # Original aura components
            'C': np.random.beta(2, 5, n_samples),  # Cellular activity
            'P': np.random.gamma(2, 2, n_samples),  # Physical/emotional state
            'F': np.random.normal(0.5, 0.2, n_samples),  # Electromagnetic field
            'E': np.random.beta(3, 3, n_samples),  # Emotional state
            'S': np.random.exponential(0.5, n_samples),  # Sensory input
            'T': np.random.lognormal(0, 0.5, n_samples),  # Duration
            'I': np.random.uniform(0, 1, n_samples),  # Individual perception
            'N': np.random.normal(0.6, 0.3, n_samples),  # Neural processing
            'M': np.random.beta(4, 2, n_samples),  # Personal beliefs
            
            # Work and efficiency metrics
            'W': np.random.gamma(3, 2, n_samples),  # Work performed
            'η': np.random.beta(5, 3, n_samples),  # Efficiency
            'Concentration': np.random.normal(0.7, 0.2, n_samples),
            'Persistence': np.random.gamma(2, 3, n_samples),
            'Strategy': np.random.beta(3, 4, n_samples),
            'Reciprocity': np.random.normal(0.5, 0.25, n_samples),
            'Trust': np.random.beta(6, 2, n_samples),
            'Impact': np.random.exponential(0.3, n_samples),
            'Meaning': np.random.normal(0.6, 0.3, n_samples),
            'Autonomy': np.random.beta(4, 3, n_samples),
        }
        
        # Normalize values to [0, 1] range
        for key in data:
            data[key] = np.clip(data[key], 0, None)
            data[key] = (data[key] - data[key].min()) / (data[key].max() - data[key].min())
        
        return pd.DataFrame(data)
    
    def calculate_derived_features(self, df):
        """Calculate derived aura features"""
        df_enhanced = df.copy()
        
        # Calculate aura components
        df_enhanced['E_a'] = self.calculate_ea(df['C'], df['P'], df['F'])
        df_enhanced['A_s'] = self.calculate_as(df['E'], df['S'], df['T'])
        df_enhanced['V_a'] = self.calculate_va(df['I'], df['C'], df['N'])
        df_enhanced['S_a'] = self.calculate_sa(df['M'], df['P'], df['E'])
        
        # Additional composite features
        df_enhanced['Aura_Intensity'] = (df_enhanced['E_a'] + df_enhanced['A_s'] + 
                                       df_enhanced['V_a'] + df_enhanced['S_a']) / 4
        
        df_enhanced['Cognitive_Index'] = (df['Concentration'] + df['Strategy'] + 
                                        df['N']) / 3
        
        df_enhanced['Emotional_Index'] = (df['E'] + df['Trust'] + df['Meaning']) / 3
        
        # Create target variable as complex function of features
        df_enhanced['Target'] = (0.3 * df_enhanced['Aura_Intensity'] + 
                               0.2 * df_enhanced['Cognitive_Index'] + 
                               0.2 * df_enhanced['Emotional_Index'] + 
                               0.15 * df['W'] + 
                               0.15 * df['η'] + 
                               np.random.normal(0, 0.05, len(df)))
        
        return df_enhanced
    
    def preprocess_data(self, df, fit_scaler=True):
        """Preprocess data with scaling and feature selection"""
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
        
        return X_selected, y
    
    def train_and_evaluate(self, df):
        """Train model and perform comprehensive evaluation"""
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
        metrics['cv_mse_mean'] = -cv_scores.mean()
        metrics['cv_mse_std'] = cv_scores.std()
        
        return metrics, (X_test, y_test, y_pred_test)
    
    def plot_results(self, X_test, y_test, y_pred_test):
        """Create visualization of results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Actual vs Predicted
        ax1.scatter(y_test, y_pred_test, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted Values')
        
        # Residuals
        residuals = y_test - y_pred_test
        ax2.scatter(y_pred_test, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = [f'Feature_{i}' for i in range(len(self.model.feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            ax3.barh(importance_df['feature'], importance_df['importance'])
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Feature Importance')
        
        # Distribution of predictions
        ax4.hist(y_pred_test, bins=20, alpha=0.7, label='Predicted')
        ax4.hist(y_test, bins=20, alpha=0.7, label='Actual')
        ax4.set_xlabel('Values')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Values')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("Initializing Enhanced Aura Analyzer...")
    analyzer = AuraAnalyzer()
    
    # Generate and process data
    print("Generating sample data...")
    df = analyzer.generate_sample_data(n_samples=1000)
    df_enhanced = analyzer.calculate_derived_features(df)
    
    print(f"Dataset shape: {df_enhanced.shape}")
    print(f"Features: {list(df_enhanced.columns)}")
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(df_enhanced.describe())
    
    # Train and evaluate model
    print("\nTraining and evaluating model...")
    metrics, test_data = analyzer.train_and_evaluate(df_enhanced)
    
    # Display results
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    X_test, y_test, y_pred_test = test_data
    analyzer.plot_results(X_test, y_test, y_pred_test)
    
    # Feature correlation analysis
    print("\nFeature Correlation Analysis:")
    correlation_matrix = df_enhanced.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()