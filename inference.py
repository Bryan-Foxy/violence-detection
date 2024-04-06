import cv2
import tensorflow as tf
import functions_tools as F
import dataset
import config

def prepare_inference(path_video, target):
    # Prepare the data for the inference

    test_ = F.frame_from_video(path_video, n_frames = 10)
    test_ = tf.expand_dims(test_, axis=0)
    target_tensor = tf.constant([target] * len(test_), dtype=tf.float32)
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
    # This function load video, write the prediction and save the new video 

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

    # Loop through the frames of the video and write the prediction
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Write the prediction on the frame
        if prediction == 1:
            text = "Violent"
            color = (0, 0, 255)  # Red for "Violent"
        else:
            text = "No Violent"
            color = (0, 255, 0)  # Green for "No Violent"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Write the frame with the prediction to the output video
        out.write(frame)

    # Close the video streams
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # Dataset
    video_paths, targets = dataset.load_video_path()
    video_dataset = dataset.load_video_and_transform(video_paths)
    train_dataset, test_dataset, train_y_dataset, test_y_dataset = dataset.split_data(video_dataset, targets)
    train_loader, test_loader = dataset.create_loader(train_dataset, test_dataset, train_y_dataset, test_y_dataset)

    # CNN-LSTM Hybrid model
    # Load the model 
    hybrid_model = tf.keras.models.load_model('saves/checkpoint/last.h5')

    # Evaluation
    loss, acc = hybrid_model.evaluate(test_loader, verbose = 2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    # We will make a prediction on one video
    target = 0
    path_video = '../video_data/real life violence situations/Real Life Violence Dataset/NonViolence/NV_237.mp4' 

    # Make the inference
    test = prepare_inference(path_video, target)
    preds = prediction(hybrid_model, test)
    cls = classes(preds)
    print('The prediction is : {} ==> {}'.format(preds, cls))

    # Make plot
    output_path = 'saves/outputs/predicted_video.avi'
    write_prediction_on_video(path_video, preds[0], output_path)
