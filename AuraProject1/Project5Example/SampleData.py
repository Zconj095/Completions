import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Sample data with varying sequence lengths
X = [
    [0, 1, 1],  # Example sequences of user actions
    [0, 1, 2],
    [0, 1, 2, 3]
]

# Padding sequences for consistent input size
X_padded = pad_sequences(X, padding='post')

# Labels for the sequences (next action prediction)
y = np.array([1, 2, 3])

# Define the model
model = Sequential([
    Embedding(input_dim=5, output_dim=2, input_length=X_padded.shape[1]),  # Adjust input_dim based on data
    LSTM(8),
    Dense(4, activation='softmax')  # Output layer for 4 possible actions
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_padded, y, epochs=10)

# Predicting the next action
test_sequence = np.array([[0, 1, 2]])
test_sequence_padded = pad_sequences(test_sequence, padding='post', maxlen=X_padded.shape[1])
prediction = model.predict(test_sequence_padded)

print(f"Predicted next action: {np.argmax(prediction, axis=1)[0]}")
