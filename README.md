# composite-damage-prediction
Surrogate model for generation of damage patterns on composite plates with cut-outs

# Contents of Repository
- data-generation
  - abaqus-models 
    > Generated Abaqus models including .inp files and .odb files, as well as README describing naming conventions
  - damage-patterns
    > Damage pattern images generated by processing .odb files
    - processed
      > Damage pattern images with borders trimmed, as well as README describing naming conventions
    - raw
      > Damage pattern images as generated by result_generator.py, as well as README describing naming conventions
    - process.m
      > Calls `RemoveWhiteSpace` on a batch of images and saves processed images in new folder
    - RemoveWhiteSpace.m
      > Trims borders of an image with padding
    - selectimagesforprocess.m
      > Determines which images in the directory have not been processed already and sends them to `process`
  - mesh_convergence.xls
    > Analysis of errors between successive FEM models to check for convergence of mesh
  - model_generator.py
    > To be attached as script in Abaqus to automate creation of Abaqus models (including .inp files) and subsequently writes to runfile
  - result_generator.py
    > To be attached as script in Abaqus to automate processing of .odb files and subsequently save damage patterns as images
  - runfile.txt
    > To be saved and run as .bat file after being modified by model_generator.py in order to generate .odb files from .inp files
- neural-network-models
  - convolutional
    > Convolutional neural network model (Python libraries, datasets and final weights not included in repo due to file size constraints)
    - tuning1
      > Training curves, predictions and log of hyperparameters used in first instance of training
    - tuning2
      > Training curves, predictions and log of hyperparameters used in second instance of training (with larger dataset)
    - cnn_architecture.py
      > Definitions of different CNN architecture variants
    - generate_data.py
      > Convert images into useable training, validation and test sets
    - helpers.py
      > Definitions of functions implemented in generate_data.py, cnn_architecture.py and main.py
    - main.py
      > Trains, validates and tests neural network
  - hybrid-mse
    - CNN_Architecture
    - generateData
    - helpers
    - input2image
  - hybrid-ssim
    - CNN_Architecture
    - generateData
    - helpers
    - input2image_ssim
  - reduced-image
    - tuning
    - CNN_Architecture
    - generateData
    - helpers
    - input2index
    - splitNetwork
  - standard
    > Standard neural network model (Python libraries, datasets and final weights not included in repo due to file size constraints)
    - tuning1
      > Training curves, predictions and log of hyperparameters used in first instance of training
    - tuning2
      > Training curves, predictions and log of hyperparameters used in second instance of training (with larger dataset)
    - generate_data.py
      > Convert images into useable training, validation and test sets
    - helpers.py
      > Definitions of functions implemented in generate_data.py and helpers.py
    - main.py
      > Trains, validates and tests neural network
    - orig_dims.py
      > Original dimensions of images - used to convert predicted output into image
- LICENSE
- README.md
- thesis.pdf
