import tensorflow as tf
import config

# Define input shape
input_shape = config.input_shape

# Define input layer
inputs = tf.keras.layers.Input(shape=input_shape)

# Create the model
model = tf.keras.Sequential([
    inputs,
    tf.keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling3D(),
    tf.keras.layers.Conv3D(64, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling3D(),
    tf.keras.layers.Conv3D(128, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling3D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=True)), # Add LSTM layer
    tf.keras.layers.Dropout(0.3), # Add dropout layer after LSTM
    tf.keras.layers.Dense(256, activation='relu'), # Dense layer with ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid') # Output layer with sigmoid activation for binary classification
])

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),  # Use binary cross-entropy for binary classification
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"]
)

# Display the model summary
model.summary()
