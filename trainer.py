from models import CNN_LSTM


if __name__ == '__main__':
    cnn_lstm = CNN_LSTM()
    input_shape = (None, 10, 224, 224, 3)  # Par exemple, pour une séquence de 10 images de taille 224x224 avec 3 canaux
    # Construire le modèle avec les dimensions d'entrée spécifiées
    cnn_lstm.build(input_shape)
    # Compile the model
    cnn_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Print model summary
    cnn_lstm.summary()