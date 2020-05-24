This directory comprises of the following folders:

- abaqus-models: All Abaqus models are present here

- damage-patterns: Image results generated from Abaqus models are saved here

The following files are also present:

- mesh_convergence.xls: Analysis of the errors between successive models, used for studying convergence of mesh

- model_generator.py: Used to create several Abaqus models (including the .inp files of these models) in one execution by looping over the loading variables, and also generate the runfile in the process

- result_generator.py: Used to process all .odb files in one execution by looping over the loading variables, and saving the resulting damage patterns as images

- runfile: This text file comprises of lines that the Windows command prompt can execute to run the .inp files of every model, the result of which is a .obd file for each (once the text file is populated, save it as a .bat file to execute with the Windows command prompt)

