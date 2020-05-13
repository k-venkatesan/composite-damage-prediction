This directory comprises of the following folders:

- Raw: Damage pattern images obtained directly from Abaqus
- Processed: Outputs of raw damage pattern images being processed to remove white space bordering them

The following files are also present:

- RemoveWhiteSpace.m: Matlab file that removes bordering white space from an image

- process.m: Matlab file that calls the 'RemoveWhiteSpace' function and saves the processed image result as a new file

- selectimagesforprocess.m: Matlab file that loops over the loading variables, damage modes and ply numbers to select images on whom the 'process' function is called