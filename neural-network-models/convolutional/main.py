import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from cnn_architecture import create_model, create_upsamp_model

# Timing the program
start_time = time.time()

# For naming the generated plots/outputs
model_num = 127

# Loading the data sets - for now, Y_set is used as both input and output - already randomized
Y_train = np.load('Y_train.npy')
Y_valid = np.load('Y_valid.npy')
Y_test = np.load('Y_test.npy') # 1763rd example

# Defining the architecture and loading weights if applicable
model = create_upsamp_model()
model.load_weights('conv_weights_more_data_5')

# Establishing training strategy
model.compile(optimizer = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999),
              loss='mse')

# Training model and storing history for plotting
trainingData = model.fit(Y_train, Y_train, epochs=400, batch_size=8, validation_data=(Y_valid, Y_valid))#, callbacks = [tb])
training_loss = trainingData.history['loss']
validation_loss = trainingData.history['val_loss']

# Evaluating and displaying the loss for the trained weights
train_set_loss = model.evaluate(Y_train, Y_train, batch_size=32)
valid_set_loss = model.evaluate(Y_valid, Y_valid, batch_size=32)
test_set_loss = model.evaluate(Y_test, Y_test, batch_size=32)
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
output = model.predict(Y_test)

# Displaying predicted output image
output_trimmed = output[:, 2:222, 2:223, :]
output_trimmed = np.reshape(output_trimmed, [220, 221, 3])
plt.imshow(output_trimmed)
plt.axis('off')
# Exact directory might be required below
plt.savefig('\Tuning\Prediction'  + str(model_num) + '.png', bbox_inches='tight')
plt.show()

# Saving weights if necessary
weights = model.save_weights('conv_weights_more_data_5')