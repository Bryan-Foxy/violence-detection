from models import build_model
import config
import dataset
import tensorflow as tf


if __name__ == '__main__':
    # Prepare the data
    video_paths, targets = dataset.load_video_path()
    video_dataset = dataset.load_video_and_transform(video_paths)
    train_dataset, test_dataset, train_y_dataset, test_y_dataset = dataset.split_data(video_dataset, targets)
    train_loader, test_loader = dataset.create_loader(train_dataset, test_dataset, train_y_dataset, test_y_dataset)

    # Build the model
    cnn_lstm, img_output = build_model(config.input_shape)
    hybrid_model = tf.keras.Model(inputs = img_output, outputs = [cnn_lstm, img_output])
    print(hybrid_model.summary())

    hybrid_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = config.lr), metrics=['accuracy','accuracy']) # Compile the model

    # Training
    history = hybrid_model.fit(train_loader,
                    steps_per_epoch = config.steps_per_epoch, validation_data = test_loader, validation_steps = config.validation_steps, epochs=config.epochs)



    