import tensorflow as tf
import keras as keras

# Generate or load image dataset
images = [] 

# Utility to animate between plots over time
def animate_plots(plot_sequence):
    anim = keras.animations.MovieWriter(fps=10) 
    for i in range(len(plot_sequence)):
        anim.add(plot_sequence[i]) 
    anim.save("quantum_simulation.gif")

# Convolutional LSTM model
conv_lstm = keras.Sequential()
conv_lstm.add(keras.layers.ConvLSTM2D(32, (3, 3), 
                 activation='relu', 
                 input_shape=(None, 128, 128, 1)))
conv_lstm.add(keras.layers.Dense(10, activation='softmax'))

# Compile 
conv_lstm.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['acc'])
                 
# Data generator to pull slices from animations               
data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

frames = data_gen.flow_from_directory('C:/Users/HeadAdminKiriguya/Desktop/New folder/images')



# Train model                  
conv_lstm.fit(frames, shuffle=True, epochs=10)

# Analysis, forecasting, etc next...