from tensorflow.keras import layers
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

# Converting images from system to data sets - each image is converted to a horizontal row and stacked vertically
def load_images(model_params, load_steps):

    # Dimensions for plate and hole in centimetres (whole major and minor axis, not half)
    L = model_params[0]
    W = model_params[1]
    t_lam = model_params[2]
    Sa = model_params[3]
    Sb = model_params[4]

    # No. of different non-zero loading conditions in X and Y direction
    nx = load_steps[0]
    ny = load_steps[1]

    # Load variable initialisation - the actual load in centimetres is the load variable divided by 100
    x = 0
    y = 0

    # Setting operating folder to where the images are present - exact directory might be required depending on installation
    folder = r'\data-generation\damage-patterns\processed\\'

    # Initializing input and output data sets - lists are faster in loops than numpy arrays
    X = []
    Y = []

    # Loading data sets for all combinations of x and y loading
    while x <= 100:

        while y <= 100:

            model_name = 'Composite_L' + str(L) + '_W' + str(W) + '_t' + str(t_lam) + '_Sa' + str(Sa) + '_Sb' + str(Sb) \
                        + '_X' + '%03d' % x + '_Y' + '%03d' % y

            # For 4 plies
            for i in range(1, 5):

                # Fibre-Compression
                image_name = 'FC_Ply' + str(i) + '_' + model_name
                file_name = folder + image_name + '.png'
                image = mpimg.imread(file_name)
                Ym = np.reshape(image, (1, image.size))
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 1]
                X.append(Xm)

                # Fibre-Tension
                image_name = 'FT_Ply' + str(i) + '_' + model_name
                file_name = folder + image_name + '.png'
                image = mpimg.imread(file_name)
                Ym = np.reshape(image, (1, image.size))
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 2]
                X.append(Xm)

                # Matrix-Compression
                image_name = 'MC_Ply' + str(i) + '_' + model_name
                file_name = folder + image_name + '.png'
                image = mpimg.imread(file_name)
                Ym = np.reshape(image, (1, image.size))
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 3]
                X.append(Xm)

                # Matrix-Tension
                image_name = 'MT_Ply' + str(i) + '_' + model_name
                file_name = folder + image_name + '.png'
                image = mpimg.imread(file_name)
                Ym = np.reshape(image, (1, image.size))
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 4]
                X.append(Xm)

            y = y + 100/ny

        y = 0
        x = x + 100/nx

    # Convert to numpy array before returning
    return np.asarray(X), np.squeeze(np.asarray(Y)), image.shape

def create_model(layer_sizes, layer_activations, weights_initializer, regularization):

    # Extracting the total number of layers (including input)
    num_layers = len(layer_sizes)

    # Instantialization of model, and creation of first hidden layer (which is supplied with information about the input)
    model = tf.keras.Sequential()
    model.add(layers.Dense(input_shape = (layer_sizes[0],),
                           units = layer_sizes[1],
                           kernel_initializer = weights_initializer,
                           kernel_regularizer = regularization,
                           use_bias = True))
    model.add(layers.Activation(layer_activations[1]))

    # Creation of other layers, including the output
    for i in range(2, num_layers):

        model.add(layers.Dense(units = layer_sizes[i],
                               kernel_initializer = weights_initializer,
                               kernel_regularizer = regularization))
        model.add(layers.Activation(layer_activations[i]))

    return model