This folder comprises of damage pattern images whose bordering white spaces have been trimmed to reduce size. The names of the files indicate the following details of the FE model:

- Mode of damage: FC (fibre-compression), FT (fibre-tension), MC (matrix-compression), or MT (matrix-tension)

- Ply number, where 1 is outer-most and 4 is closest to the mid-plane

- Length (L), in centimeteres

- Width (W), in centimetres
- Thickness (t), in centimetres

- Horizontal width of elliptical hole (Sa), in centimetres

- Vertical width of elliptical hole (Sb), in centimetres

- Horizontal displacement load (X), in 100s of centimetres*

- Vertical displacement load (Y), in 100s of centimetres* 

*That is, divide this number by 100 to obtain the displacement load in centimetres

The empty folders that exist for each of these models act as a flag - while trimming these images, a script checks if the trimmed images for a particular model is already existing by checking for the empty folder, and executes the commands to do so only if it isn't the case.