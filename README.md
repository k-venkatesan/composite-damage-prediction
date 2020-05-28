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
    - tuning1
    - tuning2
    - CNN_Architecture.py
    - generateData.py
    - helpers.py
    - main.py
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
    - tuning1
    - tuning2
    - generateData.py
    - helpers.py
    - main.py
    - origDims.py
- LICENSE
- README.md
- thesis.pdf
