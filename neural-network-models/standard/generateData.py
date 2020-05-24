from helpers import load_images
import numpy as np

# Dimensions for plate and hole in centimetres (whole major and minor axis, not half)
L = 10
W = 10
t_lam = 1
Sa = 2
Sb = 4
# No. of different non-zero loading conditions in X and Y direction
nx = 20
ny = 20
# Putting parameters in lists to feed to the load_images() function
modelParams = [L, W, t_lam, Sa, Sb]
loadSteps = [nx, ny]

# Creating datasets and storing them as a file for loading during use - origDims are the original image dimensions,
# required for converting the predicted outputs from a flattened array to an image
X_set, Y_set, origDims = load_images(modelParams, loadSteps)

X_test = np.reshape(X_set[1763], [1, 4])
Y_test = np.reshape(Y_set[1763], [1, 145860])

X_remaining = np.concatenate((X_set[0:1763], X_set[1764:7056]), axis = 0)
Y_remaining = np.concatenate((Y_set[0:1763], Y_set[1764:7056]), axis = 0)

shuffle = np.random.permutation(7055)
X_remaining_shuffled = X_remaining[shuffle]
Y_remaining_shuffled = Y_remaining[shuffle]

X_train, Y_train = X_remaining_shuffled[0:5600], Y_remaining_shuffled[0:5600]
X_valid, Y_valid = X_remaining_shuffled[5600:6400], Y_remaining_shuffled[5600:6400]
X_reserve, Y_reserve = X_remaining_shuffled[6400:7055], Y_remaining_shuffled[6400:7055]

np.save('X_train.npy', X_train)
np.save('X_valid.npy', X_valid)
np.save('X_test.npy', X_test)
np.save('X_reserve.npy', X_reserve)
np.save('Y_train.npy', Y_train)
np.save('Y_valid.npy', Y_valid)
np.save('Y_test.npy', Y_test)
np.save('Y_reserve.npy', Y_reserve)
np.save('origDims.npy', origDims)

