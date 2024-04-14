import argparse
import cv2
import tensorflow as tf
import functions_tools as F
import config

def prepare_inference(path_video, target):
    # Prepare the data for inference

    test_ = F.frame_from_video(path_video, n_frames=10)
    test_ = tf.expand_dims(test_, axis=0)
    target_tensor = tf.constant([target] * len(test_), dtype=tf.float32)
    test_ = test_ / 255.0 # Suppose pixel are 2⁸
    test = tf.data.Dataset.from_tensor_slices((test_, target_tensor)).batch(config.batch).cache().prefetch(tf.data.AUTOTUNE)

    return test
    

def prediction(model, test):
    # Make prediction

    preds_prob = model.predict(test)
    preds = [1 if prediction > 0.5 else 0 for prediction in preds_prob] 
    return preds 

def classes(preds):
    # Return true classes

    cls = ['Violence' if pred > 0.5 else 'No Violence' for pred in preds]
    return cls

def write_prediction_on_video(video_path, prediction, output_path):
    # This function loads the video, writes the prediction, and saves the new video

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video.")
        return

    # Get the dimensions of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a writer to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    # Define the position and size of the box (adjusted for top-left corner and reduced size)
    box_x, box_y = 10, 10  # Top-left corner position of the box
    box_width, box_height = 100, 30  # Reduced width and height of the box

    # Loop through the frames of the video and write the prediction
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the colored box as background for the prediction text
        if prediction == 1:
            box_color = (0, 0, 255)  # Red for "Violent"
        else:
            box_color = (0, 255, 0)  # Green for "No Violent"
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), box_color, -1)  # Draw filled rectangle

        # Write the prediction on the frame
        text = "Violent" if prediction == 1 else "No Violent"
        cv2.putText(frame, text, (box_x + 5, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

        # Write the frame with the prediction to the output video
        out.write(frame)

    # Close the video streams
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main(args):
    # Load the model
    hybrid_model = tf.keras.models.load_model('saves/checkpoint/model.keras')

    # We will make a prediction on the provided video
    target = 0

    # Make the inference
    test = prepare_inference(args.input, target)
    preds = prediction(hybrid_model, test)
    cls = classes(preds)
    print('The prediction is: {} ==> {}'.format(preds, cls))

    # Make plot
    write_prediction_on_video(args.input, preds[0], args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference on a video.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output video file.")
    args = parser.parse_args()
    main(args)
