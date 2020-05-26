# Timing the program
import time
start_time = time.time()

# For naming the generated plots/outputs
model_num = 93

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from CNN_Architecture import denseUp
from helpers import ssim_loss

# Loading datasets
X_train = np.load('X_train.npy')
X_valid = np.load('X_valid.npy')
X_test = np.load('X_test.npy')
X_reserve = np.load('X_reserve.npy')
Y_train = np.load('Y_train.npy')
Y_valid = np.load('Y_valid.npy')
Y_test = np.load('Y_test.npy')
Y_reserve = np.load('Y_reserve.npy')

# Creating model and loading weights if applicable
model = denseUp()
model.load_weights('ssim_12544_2')

# Establishing training strategy
model.compile(optimizer = tf.train.AdamOptimizer(learning_rate = 0.00003, beta1 = 0.9, beta2 = 0.999),
              loss = ssim_loss)

# Training model and storing history for later analysis
trainingData = model.fit(X_train, Y_train, epochs = 50, batch_size = 32, validation_data = (X_valid, Y_valid))
training_loss = trainingData.history['loss']
validation_loss = trainingData.history['val_loss']

# To evaluate performance, indexed vector is generated and fed to the Gen Model to compute error wrt image
train_set_loss = model.evaluate(X_train, Y_train, batch_size = 32)
valid_set_loss = model.evaluate(X_valid, Y_valid, batch_size = 32)
test_set_loss = model.evaluate(X_test, Y_test, batch_size = 32)
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

# Predicting output using trained weights for 1763rd example
output = model.predict(X_test)

# Displaying predicted output image
output_trimmed = output[:, 2:222, 2:223, :]
output_trimmed = np.reshape(output_trimmed, [220, 221, 3])
plt.imshow(output_trimmed)
plt.axis('off')
# Exact directory might be required below
plt.savefig('\Tuning\Prediction' + str(model_num) + '.png', bbox_inches = 'tight')
plt.show()

# Saving weights if necessary
model.save_weights('ssim_12544_2')