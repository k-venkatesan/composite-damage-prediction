import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from skimage.measure import compare_ssim

# Converting images from system to data sets - each image is padded according to the array 'paddings'
def load_images(modelParams, loadSteps, paddings):

    # Dimensions for plate and hole in centimetres (whole major and minor axis, not half)
    L = modelParams[0]
    W = modelParams[1]
    t_lam = modelParams[2]
    Sa = modelParams[3]
    Sb = modelParams[4]
    # No. of different non-zero loading conditions in X and Y direction
    nx = loadSteps[0]
    ny = loadSteps[1]
    # Load variable initialisation - the actual load in centimetres is the load variable divided by 100
    x = 0
    y = 0

    # Setting operating folder to where the images are present - exact directory might be required
    folder = r'C:\Users\Karthik Venkatesan\Documents\Master Thesis\Structural Mechanics using Artificial Neural Networks\Project\Abaqus\DataGeneration\DamagePatterns\Processed\\'
    # Initializing input and output data sets - lists are faster in loops than numpy arrays
    X = []
    Y = []

    # Loading data sets for all combinations of x and y loading
    while x <= 100:

        while y <= 100:
            modelName = 'Composite_L' + str(L) + '_W' + str(W) + '_t' + str(t_lam) + '_Sa' + str(Sa) + '_Sb' + str(Sb) \
                        + '_X' + '%03d' % x + '_Y' + '%03d' % y

            # For 4 plies
            for i in range(1, 5):

                # Fibre-Compression
                imageName = 'FC_Ply' + str(i) + '_' + modelName
                fileName = folder + imageName + '.png'
                Ym = mpimg.imread(fileName)
                Ym = np.pad(Ym, paddings, 'constant')
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 1]
                X.append(Xm)

                # Fibre-Tension
                imageName = 'FT_Ply' + str(i) + '_' + modelName
                fileName = folder + imageName + '.png'
                Ym = mpimg.imread(fileName)
                Ym = np.pad(Ym, paddings, 'constant')
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 2]
                X.append(Xm)

                # Matrix-Compression
                imageName = 'MC_Ply' + str(i) + '_' + modelName
                fileName = folder + imageName + '.png'
                Ym = mpimg.imread(fileName)
                Ym = np.pad(Ym, paddings, 'constant')
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 3]
                X.append(Xm)

                # Matrix-Tension
                imageName = 'MT_Ply' + str(i) + '_' + modelName
                fileName = folder + imageName + '.png'
                Ym = mpimg.imread(fileName)
                Ym = np.pad(Ym, paddings, 'constant')
                Ym.tolist()
                Y.append(Ym)
                Xm = [x * 0.001, y * 0.001, i, 4]
                X.append(Xm)

            y = y + 100/ny

        y = 0
        x = x + 100/nx

    # Convert to numpy array before returning
    return np.asarray(X), np.squeeze(np.asarray(Y))

# Creating of standard NN architecture
def createDenseModel(layer_sizes, layer_activations, weights_initializer, regularization):

    # Extracting the total number of layers (including input)
    num_layers = len(layer_sizes)
    # Instantialization of model, and creation of first hidden layer
    # (which is supplied with information about the input)
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

# Defining loss function to be minimised as 1-SSIM
def ssim_loss(truth, prediction):
    return 1 - tf.image.ssim(truth, prediction, max_val=1.0)