# Timing the program
import time
start_time = time.time()

# For naming the generated plots/outputs
model_num = 147

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from helpers import createModel

# Loading datasets - orig_dims are the original image dimensions, required for converting the predicted outputs from a
# flattened array to an image
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_valid = np.load('X_valid.npy')
Y_valid = np.load('Y_valid.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
orig_dims = np.load('orig_dims.npy')

# Definition of network architecture
input_size = [X_train.shape[1]]
output_size = [Y_train.shape[1]]
hidden_layer_sizes = [250, 250, 250]
hidden_layer_activations = ['relu', 'relu', 'relu']
output_layer_activation = ['sigmoid']
weights_initializer = 'glorot_uniform'
regularization = None

# Concatenating lists for ease of feeding into function for model creation
layer_sizes = input_size + hidden_layer_sizes + output_size
layer_activations = ['null'] + hidden_layer_activations + output_layer_activation

# Creation of model object
model = createModel(layer_sizes, layer_activations, weights_initializer, regularization)

# Loading trained weights, if applicable
model.load_weights('weights')


# Establishing training strategy
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999),
              loss='mse')

# Training model and storing history for later analysis
training_data = model.fit(X_train, Y_train, epochs=2000, batch_size=32, validation_data=(X_valid, Y_valid))
training_loss = training_data.history['loss']
validation_loss = training_data.history['val_loss']

# Evaluating and displaying the loss for the trained weights
train_set_loss = model.evaluate(X_train, Y_train, batch_size=32)
valid_set_loss = model.evaluate(X_valid, Y_valid, batch_size=32)
test_set_loss = model.evaluate(X_test, Y_test, batch_size=32)
print('Training loss: ' + str(train_set_loss))
print('Validation loss: ' + str(valid_set_loss))
print('Test loss: ' + str(test_set_loss))

# Printing the total run time
duration = time.time() - start_time
print('Run time: ' + str(duration))

# Visualizing loss history
epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss)#, 'r--')
plt.plot(epoch_count, validation_loss)#, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
# Exact directory might be required below
plt.savefig('\Tuning2\Loss' + str(model_num) + '.png', bbox_inches='tight')
plt.show()

# Predicting output using trained weights for 100th example
output = model.predict(X_test)

# Displaying predicted output image
output = np.reshape(output, orig_dims)
plt.imshow(output)
plt.axis('off')
# Exact directory might be required below
plt.savefig('\Tuning\Prediction'  + str(model_num) + '.png', bbox_inches='tight')
plt.show()

# Saving weights if required
model.save_weights('weights')