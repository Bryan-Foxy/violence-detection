import gradio as gr
import tensorflow as tf
import functions_tools as F
import config
import inference
import cv2
import numpy as np
import os

# Load the model
model = tf.keras.models.load_model('saves/checkpoint/model.keras')


def predict_violence(video_path):
    # Extract the video's name without the extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Prepare the data for the inference
    test_ = F.frame_from_video(video_path, n_frames=10)
    test_ = tf.expand_dims(test_, axis=0)
    target_tensor = tf.constant([0] * len(test_), dtype=tf.float32)
    test = tf.data.Dataset.from_tensor_slices((test_, target_tensor)).batch(config.batch).cache().prefetch(
        tf.data.AUTOTUNE)

    # Make prediction
    preds_prob = model.predict(test)
    preds = [1 if prediction > 0.5 else 0 for prediction in preds_prob]

    # Generate a unique output path based on the video's name
    output_dir = 'saves/outputs'
    output_path = os.path.join(output_dir, f"{video_name}_predicted.avi")

    # Write the prediction on the video
    inference.write_prediction_on_video(video_path, preds[0], output_path)

    # Return the video file with the prediction written on it
    return output_path


# Define the Gradio interface
iface = gr.Interface(
    fn=predict_violence,
    inputs=gr.Video(label="Upload a video"),
    outputs=gr.Video(),
    title="Violence Detection",
    description="Upload a video to detect violence."
)

# Launch the interface
iface.launch()