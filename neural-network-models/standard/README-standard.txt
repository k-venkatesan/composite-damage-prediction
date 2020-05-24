This directory comprises of the following folders:

- tuning1: The results of the first hyperparameter tuning process of the standard neural network are available here

- tuning2: The results of another later hyperparameter tuning process of the standard neural network are available here

The following files are also present:

- main.py: Main python file that sets up and runs the neural network

- helpers.py: Definition of user-defined functions called in main.py

- generateData.py: Script used to generate datasets from the damage pattern images

- origDims.npy: Original dimensions of the image, used in main.py

- weights of training neural network

- datasets for X (input) and Y (output), segregated from entire set as training, validation, test and reserve sets

What is NOT present:

- Libraries and python packages used in the code. Remember to download these to your python environment before running the code. Install 'tensorflow-gpu' instead of 'tensorflow' for efficiency.