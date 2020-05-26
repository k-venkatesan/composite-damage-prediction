# Timing the program
import time
start_time = time.time()

# For naming the generated plots/outputs
model_num = 277

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from helpers import createDenseModel
from CNN_Architecture import createUpSampModel, genImages

X_train = np.load('X_train.npy')
X_valid = np.load('X_valid.npy')
X_test = np.load('X_test.npy')
Y_train = np.load('Y_train.npy')
Y_valid = np.load('Y_valid.npy')
Y_test = np.load('Y_test.npy')
indexedImages_train = np.load('indexedImages_train.npy')
indexedImages_valid = np.load('indexedImages_valid.npy')
indexedImages_test = np.reshape(np.load('indexedImages_test.npy'), [1, 6272])

# Definition of network architecture
input_size = [X_train.shape[1]]
output_size = [indexedImages_train.shape[1]]
hidden_layer_sizes = [250, 250, 250]
hidden_layer_activations = ['relu', 'relu', 'relu', 'relu']
output_layer_activation = ['relu']
weights_initializer = 'glorot_uniform'
regularization = None
# Concatenating lists for ease of feeding into function for model creation
layer_sizes = input_size + hidden_layer_sizes + output_size
layer_activations = ['null'] + hidden_layer_activations + output_layer_activation

# Creation of model object
model = createDenseModel(layer_sizes, layer_activations, weights_initializer, regularization)
model.load_weights('weights_input2index')

# Establishing training strategy
model.compile(optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999),
              loss = 'mse')

# Training model and storing history for later analysis
trainingData = model.fit(X_train, indexedImages_train, epochs = 1000, batch_size = 16, validation_data = (X_valid, indexedImages_valid))
training_loss = trainingData.history['loss']
validation_loss = trainingData.history['val_loss']

# To evaluate performance, indexed vector is generated and fed to the Gen Model to compute error wrt image
train_set_predict = np.reshape(model.predict(X_train), [5600, 1, 6272])
valid_set_predict = np.reshape(model.predict(X_valid), [800, 1, 6272])
test_set_predict = np.reshape(model.predict(X_test), [1, 1, 6272])
# Creation of CNN model with trained weights, that will be split into networks for generating feature index and
# converting feature index to image
usModel = createUpSampModel()
usModel.load_weights('conv_weights_more_data_5')
genModel = genImages(usModel)
genModel.compile(optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999), loss = 'mse')
train_set_loss = genModel.evaluate(train_set_predict, Y_train, batch_size=32)
valid_set_loss = genModel.evaluate(valid_set_predict, Y_valid, batch_size=32)
test_set_loss = genModel.evaluate(test_set_predict, Y_test, batch_size=32)
print('Training loss: ' + str(train_set_loss))
print('Validation loss: ' + str(valid_set_loss))
print('Test loss: ' + str(test_set_loss))

# Printing the total run time
duration = time.time() - start_time
print('Run time: ' + str(duration))

# Visualizing loss history - this loss is only for input to index, not image
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss)#, 'r--')
plt.plot(epoch_count, validation_loss)#, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
# Exact directory might be required below
plt.savefig('\Tuning\Loss' + str(model_num) + '.png', bbox_inches = 'tight')
plt.show()

# Predicting output using trained weights for 100th example
output = genModel.predict(np.reshape(test_set_predict, [1, 1, 6272]))

# Displaying predicted output image
output_trimmed = output[:, 2:222, 2:223, :]
output_trimmed = np.reshape(output_trimmed, [220, 221, 3])
plt.imshow(output_trimmed)
plt.axis('off')
# Exact directory might be required below
plt.savefig('\Tuning\Prediction' + str(model_num) + '.png', bbox_inches = 'tight')
plt.show()

# Saving weights if necessary
model.save_weights('weights_input2index_more_2')