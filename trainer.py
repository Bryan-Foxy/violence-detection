from c3d import C3D
from sklearn.metrics import classification_report
import config
import dataset
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    # Prepare the data
    video_paths, targets = dataset.load_video_path()
    video_dataset = dataset.load_video_and_transform(video_paths)
    train_dataset, test_dataset, train_y_dataset, test_y_dataset = dataset.split_data(video_dataset, targets)
    train_loader, test_loader = dataset.create_loader(train_dataset, test_dataset, train_y_dataset, test_y_dataset)

    # Build the model
    c3d, img_output = C3D(config.input_shape)
    c3d_model = tf.keras.Model(inputs = img_output, outputs = [c3d], name = 'C3D')
    print(c3d_model.summary())


    c3d_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = config.lr), metrics=['accuracy']) # Compile the model

    # Training
    history = c3d_model.fit(train_loader,
                    validation_data = test_loader, 
                    epochs=config.epochs,
                    callbacks=[config.early_stopping, config.model_checkpoint, config.tensorboard_callback])
    
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
    plt.legend()
    plt.show()
    
    # Save the model after the training
    c3d_model.save('saves/checkpoint/last.h5')

    # Evaluation
    predictions = c3d_model.predict(test_loader)
    predictions = [1 if pred > 0.5 else 0 for pred in predictions]
    print(classification_report(test_y_dataset, predictions))



    