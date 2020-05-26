This directory comprises of the following folders:

- Tuning1: The results of the first and then later final hyperparameter tuning process of the convolutional neural network are available here

- Tuning2: The results of an in-between hyperparameter tuning process of the convolutional neural network are available here

The following files are also present:

- main.py: Main python file that sets up and runs the neural network

- helpers.py: Definition of user-defined functions called in main.py

- CNN_Architecture.py: Defintion of architecture of CNN

- generateData.py: Script used to generate datasets from the damage pattern images

- origDims.npy: Original dimensions of the image, used in main.py

- weights of training neural network

- datasets for X (input) and Y (output), segregated from entire set as training, validation, test and reserve sets

What is NOT present:

- Libraries and python packages used in the code. Remember to download these to your python environment before running the code. Install 'tensorflow-gpu' instead of 'tensorflow' for efficiency.