from cnn_lstm import build_model
from sklearn.metrics import classification_report
import time
import config
import dataset
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model


if __name__ == '__main__':
    # Prepare the data
    video_paths, targets = dataset.load_video_path()
    video_dataset = dataset.load_video_and_transform(video_paths)
    train_dataset, test_dataset, train_y_dataset, test_y_dataset = dataset.split_data(video_dataset, targets)
    train_dataset, test_dataset = dataset.normalize(train_dataset, test_dataset)
    train_loader, test_loader = dataset.create_loader(train_dataset, test_dataset, train_y_dataset, test_y_dataset)

    # Build the model
    hybrid, img_output = build_model(config.input_shape)
    hybrid_model = tf.keras.Model(inputs = img_output, outputs = [hybrid], name = 'C3D_LSTM')
    print(hybrid_model.summary())

    # Plot the architecture
    plot_model(hybrid_model, to_file='C3D_LSTM.png', show_shapes=True, show_layer_names=True)

    hybrid_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), metrics=['accuracy']) # Compile the model

    # Training
    start = time.time()
    history = hybrid_model.fit(train_loader,
                    validation_data = test_loader, 
                    epochs=config.epochs,
                    callbacks=[config.early_stopping, config.model_checkpoint, config.tensorboard_callback])
    
    end = time.time()
    print('The model takes {:.3f}min to train'.format((end-start)/60))
    # History
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(train_loss) + 1)

    # Figure
    plt.plot(epochs_range, train_loss, 'b', label='Training loss')
    plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2, fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Save the model after the training
    hybrid_model.save('saves/checkpoint/model.keras')

    # Evaluation
    predictions = hybrid_model.predict(test_loader)
    predictions = [1 if pred > 0.5 else 0 for pred in predictions]
    report = classification_report(test_y_dataset, predictions)
    print(report)
    
    with open('classification_report.txt', 'w') as file:
        file.write("Classification Report:\n")
        file.write(report + "\n")

        file.write("Predictions:\n")
        for pred, actual in zip(predictions, test_y_dataset):
            file.write(f"Predicted: {pred}, Actual: {actual}\n")



    