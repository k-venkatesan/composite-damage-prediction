This directory comprises of the following folders:

- Tuning: The results of the hyperparameter tuning process of the reduced-image neural network are available here

The following files are also present:

- input2index.py: Main python file that sets up and runs the neural network

- splitNetwork.py: Script that splits a CNN model in two one that generates the feature index for an image, and one that outputs a predicted image from a feature index

- helpers.py: Definition of user-defined functions called in main.py

- CNN_Architecture.py: Defintion of architecture of CNN

- generateData.py: Script used to generate datasets from the damage pattern images

- origDims.npy: Original dimensions of the image, used in main.py

- feature indices of images in training, validation and test sets

- weights of the CNN that are used are convolving an image to a feature index, and deconvolving a feature index to a predicted damage pattern

- weights of training the reduced-image neural network

What is NOT present:

- Libraries and python packages used in the code. Remember to download these to your python environment before running the code. Install 'tensorflow-gpu' instead of 'tensorflow' for efficiency.

- X and Y datasets - they are the same files as in the 'Convolutional' folder