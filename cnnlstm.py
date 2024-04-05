import tensorflow as tf

def spatial_block(input, filters, strides, pool_size, pool_stride):
    # Build the block to extract spatial convolution

    x = tf.keras.layers.Conv3D(filters, strides, padding = 'same')(input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling3D(pool_size = pool_size, strides = pool_stride, padding = 'same')(x)
    x = tf.keras.layers.Conv3D(filters, strides, padding = 'same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPooling3D(pool_size = pool_size, strides = pool_stride, padding = 'same')(x)
    return x 

def temporal_block(input, units, filter):
    # Build block to extract temporal information

    x = tf.keras.layers.ConvLSTM2D(units, filter, dropout = 0.2)(input)
    x = tf.keras.layers.Flatten()(x)
    return x 

def build_model(input):
    # Build the CNN-LSTM hybrid model to extract spatio-temporal information
    # Input
    img_input = tf.keras.layers.Input(shape=input)
    #padded_input = tf.keras.layers.ZeroPadding3D(padding=((0,6), (0,6), (0,0)))(img_input)
    # Spatial Block
    x = spatial_block(img_input, 32, strides = (3,3,3), pool_size = (2,2,2), pool_stride = (2,2,2))
    x = spatial_block(x, 64, strides = (3,3,3), pool_size = (2,2,2), pool_stride = (2,2,2))
    x = spatial_block(x, 128, strides = (3,3,3), pool_size = (2,2,2), pool_stride = (2,2,2))
    x = tf.keras.layers.BatchNormalization()(x)
    # Temporal Block
    x = temporal_block(x, units = 256, filter = (3,3))

    # MultiLayer Perceptron
    x = tf.keras.layers.Dense(units = 128)(x)
    x = tf.keras.layers.LeakyReLU()(x) 
    x = tf.keras.layers.Dense(1, activation='sigmoid', name = 'video_cls')(x)
    return x, img_input

