import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from CNN_Architecture import createUpSampModel, indexImages, genImages

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_valid = np.load('X_valid.npy')
Y_valid = np.load('Y_valid.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

model = createUpSampModel()
model.load_weights('conv_weights_more_data_5')

indexModel = indexImages(model)
indexedImages_train = indexModel.predict(Y_train)
indexedImages_valid = indexModel.predict(Y_valid)
indexedImages_test = indexModel.predict(Y_test)

np.save('indexedImages_train.npy', indexedImages_train)
np.save('indexedImages_valid.npy', indexedImages_valid)
np.save('indexedImages_test.npy', np.reshape(indexedImages_test, [1, 1, 6272]))

#genModel = genImages(model)
#tf.keras.models.save_model(genModel, filepath='./genModel', include_optimizer=False)