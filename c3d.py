import tensorflow as tf

def C3D(input):
    # Build the Conv3D model to extract spatio-temporal information
    # Input
    img_input = tf.keras.layers.Input(shape=input)
    #padded_input = tf.keras.layers.ZeroPadding3D(padding=((0,6), (0,6), (0,0)))(img_input)
    x = tf.keras.layers.Conv3D(filters = 64, kernel_size = (3,3,3), padding = 'same')(img_input)
    x = tf.keras.layers.MaxPool3D(pool_size = (1,2,2), strides = (1,2,2), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(filters = 128, kernel_size = (3,3,3), padding = 'same')(x)
    x = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(filters = 256, kernel_size = (3,3,3),padding='same')(x)
    x = tf.keras.layers.Conv3D(filters = 256, kernel_size = (3,3,3),padding='same')(x)
    x = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3),padding='same')(x)
    x = tf.keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3),padding='same')(x)
    x = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3),padding='same')(x)
    x = tf.keras.layers.Conv3D(filters = 512, kernel_size = (3,3,3),padding='same')(x)
    x = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(x)

    # MultiLayer Perceptron
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.Sequential([
        tf.keras.layers.Dense(units=4096),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=4096),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(1, activation='sigmoid', name='video_cls')
    ])(x)

    return x, img_input

