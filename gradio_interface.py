import gradio as gr
from gradio import Video
import tensorflow as tf
import functions_tools as F
import config


# Load the model
model = tf.keras.models.load_model('saves/checkpoint/last.h5')

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def predict_violence(video_path):
    # Prepare the data for the inference
    test_ = F.frame_from_video(video_path, n_frames=10)
    test_ = tf.expand_dims(test_, axis=0)
    target_tensor = tf.constant([0] * len(test_), dtype=tf.float32)
    test = tf.data.Dataset.from_tensor_slices((test_, target_tensor)).batch(config.batch).cache().prefetch(tf.data.AUTOTUNE)
    
    # Make prediction
    preds_prob = model.predict(test)
    preds = [1 if prediction > 0.5 else 0 for prediction in preds_prob] 
    
    # Return the prediction
    return 'Violence' if preds[0] == 1 else 'No Violence'

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_violence, 
    inputs=Video(label="Upload a video"),
    outputs="text",
    title="Violence Detection",
    description="Upload a video to detect violence."
)

# Launch the interface
iface.launch()
