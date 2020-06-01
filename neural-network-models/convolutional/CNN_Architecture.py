import tensorflow as tf
from tensorflow.keras import layers

# Definition of CNN architecture
def create_model():

    model = tf.keras.Sequential()

    # Convolution
    model.add(layers.InputLayer(input_shape=(224, 224, 3)))
    # Current size = (224, 224, 3)
    model.add(layers.Conv2D(filters=4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2D(filters=4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2D(filters=8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 8)
    model.add(layers.Conv2D(filters=8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 8)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (112, 112, 8)
    model.add(layers.Flatten())
    # Current size = 100352

    # Deconvolution
    # Current size = 100352
    model.add(layers.Reshape(target_shape=(112, 112, 8)))
    # Current size = (112, 112, 8)
    model.add(layers.Conv2DTranspose(filters=8,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same',
                                     activation='relu'))
    # Current size = (224, 224, 8)
    model.add(layers.Conv2DTranspose(filters=4,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    #model.add(layers.BatchNormalization(gamma_initializer='glorot_uniform'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2DTranspose(filters=4,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    # model.add(layers.BatchNormalization(gamma_initializer='glorot_uniform'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2DTranspose(filters=3,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='sigmoid'))
    # Output size = (224, 224, 3)

    return model

# Definition of CNN model using upsampling
def create_upsamp_model():

    model = tf.keras.Sequential()

    # Convolution
    model.add(layers.InputLayer(input_shape=(224, 224, 3)))
    # Current size = (224, 224, 3)
    model.add(layers.Conv2D(filters=4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2D(filters=8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 8)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (112, 112, 8)
    model.add(layers.Conv2D(filters=16,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (112, 112, 16)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (56, 56, 16)
    model.add(layers.Conv2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (56, 56, 32)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (28, 28, 32)
    model.add(layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (28, 28, 64)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (14, 14, 64)
    model.add(layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (14, 14, 128)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (7, 7, 128)
    model.add(layers.Flatten(name='Indexed_Vector'))
    # Current size = 6272

    # Deconvolution
    # Current size = 6272
    model.add(layers.Reshape(target_shape=(7, 7, 128)))
    # Current size = (7, 7, 128)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (14, 14, 128)
    model.add(layers.Conv2DTranspose(filters=64,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (14, 14, 64)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (28, 28, 64)
    model.add(layers.Conv2DTranspose(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (28, 28, 32)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (56, 56, 32)
    model.add(layers.Conv2DTranspose(filters=16,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (56, 56, 16)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (112, 112, 16)
    model.add(layers.Conv2DTranspose(filters=8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (112, 112, 8)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (224, 224, 8)
    model.add(layers.Conv2DTranspose(filters=4,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2DTranspose(filters=3,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='sigmoid'))
    # Output size = (224, 224, 3)

    return model

# Model initialisation to reduce images to feature index based on convolution weights present in 'model1'
def index_images(model1):

    model = tf.keras.Sequential()

    # Convolution
    model.add(layers.InputLayer(input_shape=(224, 224, 3)))
    # Current size = (224, 224, 3)
    model.add(layers.Conv2D(filters=4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            weights=model1.layers[0].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2D(filters=8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            weights=model1.layers[2].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 8)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (112, 112, 8)
    model.add(layers.Conv2D(filters=16,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            weights=model1.layers[5].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (112, 112, 16)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (56, 56, 16)
    model.add(layers.Conv2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            weights=model1.layers[8].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (56, 56, 32)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (28, 28, 32)
    model.add(layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            weights=model1.layers[11].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (28, 28, 64)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (14, 14, 64)
    model.add(layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            weights=model1.layers[14].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (14, 14, 128)
    model.add(layers.AveragePooling2D(2, 2))
    # Current size = (7, 7, 128)
    model.add(layers.Flatten(name='Indexed_Vector'))
    # Current size = 6272

    return model

# Model initialisation to convert feature index to image using deconvolution weights from 'model1'
def gen_images(model1):

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(1, 6272)))
    # Deconvolution
    # Current size = 6272
    model.add(layers.Reshape(target_shape=(7, 7, 128)))
    # Current size = (7, 7, 128)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (14, 14, 128)
    model.add(layers.Conv2DTranspose(filters=64,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     weights=model1.layers[20].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (14, 14, 64)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (28, 28, 64)
    model.add(layers.Conv2DTranspose(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     weights=model1.layers[23].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (28, 28, 32)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (56, 56, 32)
    model.add(layers.Conv2DTranspose(filters=16,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     weights=model1.layers[26].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (56, 56, 16)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (112, 112, 16)
    model.add(layers.Conv2DTranspose(filters=8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     weights=model1.layers[29].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (112, 112, 8)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (224, 224, 8)
    model.add(layers.Conv2DTranspose(filters=4,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     weights=model1.layers[32].get_weights()))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2DTranspose(filters=3,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='sigmoid',
                                     weights=model1.layers[34].get_weights()))
    # Output size = (224, 224, 3)

    return model

# Model initialisation of Hybrid Network
def dense_up():

    model = tf.keras.Sequential()

    model.add(layers.Dense(200))
    model.add(layers.Activation('relu'))
    # Current size = 200

    model.add(layers.Dense(200))
    model.add(layers.Activation('relu'))
    # Current size = 200

    model.add(layers.Dense(12544))
    model.add(layers.Activation('relu'))
    # Current size = 12544

    model.add(layers.Reshape(target_shape=(14, 14, 64)))
    # Current size = (14, 14, 64)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (28, 28, 64)
    model.add(layers.Conv2DTranspose(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (28, 28, 32)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (56, 56, 32)
    model.add(layers.Conv2DTranspose(filters=16,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (56, 56, 16)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (112, 112, 16)
    model.add(layers.Conv2DTranspose(filters=8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (112, 112, 8)
    model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
    # Current size = (224, 224, 8)
    model.add(layers.Conv2DTranspose(filters=4,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same'))
    model.add(layers.Activation('relu'))
    # Current size = (224, 224, 4)
    model.add(layers.Conv2DTranspose(filters=3,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='sigmoid'))
    # Output size = (224, 224, 3)

    return model