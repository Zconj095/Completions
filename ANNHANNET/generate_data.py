import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing TensorFlow components
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
except ImportError as e:
    logger.error(f"TensorFlow import failed: {e}")
    raise ImportError("Please install TensorFlow: pip install tensorflow")

class HANETDataGenerator:
    """Data generator for Hybrid Ad-hoc NETwork routing optimization."""
    
    def __init__(self, num_nodes: int = 5, seed: int = 42):
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        
    def generate_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic network topology and routing data."""
        # Network adjacency matrix (flattened)
        X = np.random.rand(num_samples, self.num_nodes * self.num_nodes)
        
        # Add network features (distance matrix, node degrees, etc.)
        node_features = np.random.rand(num_samples, self.num_nodes * 3)  # Additional features
        X = np.concatenate([X, node_features], axis=1)
        
        # Target: optimal routing probabilities for each node
        y = np.random.rand(num_samples, self.num_nodes)
        y = y / y.sum(axis=1, keepdims=True)  # Normalize to probabilities
        
        return X, y

class HANETModel:
    """Neural network model for HANET routing optimization."""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_model(self) -> Sequential:
        """Build the neural network architecture."""
        model = Sequential([
            Dense(128, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(self.output_dim, activation='softmax')  # Softmax for routing probabilities
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> None:
        """Train the model with early stopping and model checkpointing."""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint('best_hanet_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance."""
        X_test_scaled = self.scaler.transform(X_test)
        loss, accuracy, mae = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        return {'loss': loss, 'accuracy': accuracy, 'mae': mae}
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].legend()
        
        # MAE
        axes[1, 0].plot(self.history.history['mae'], label='Training MAE')
        axes[1, 0].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1, 0].set_title('Mean Absolute Error')
        axes[1, 0].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function."""
    try:
        # Parameters
        NUM_SAMPLES = 5000
        NUM_NODES = 8
        TEST_SIZE = 0.2
        VAL_SIZE = 0.1
        
        logger.info("Generating synthetic HANET data...")
        data_generator = HANETDataGenerator(num_nodes=NUM_NODES)
        X, y = data_generator.generate_data(num_samples=NUM_SAMPLES)
        
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=VAL_SIZE, random_state=42
        )
        
        # Create and train model
        logger.info("Building HANET model...")
        hanet_model = HANETModel(input_dim=X.shape[1], output_dim=y.shape[1])
        hanet_model.build_model()
        
        logger.info("Training model...")
        hanet_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        
        # Evaluate model
        logger.info("Evaluating model...")
        test_results = hanet_model.evaluate(X_test, y_test)
        logger.info(f"Test Results: {test_results}")
        
        # Plot results
        hanet_model.plot_training_history()
        
        # Save final model
        hanet_model.model.save('hanet_final_model.h5')
        logger.info("Model saved successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
