from abaqus import *
from abaqusConstants import *
import __main__
import os

import visualization
import xyPlot
import displayGroupOdbToolset as dgo

# Changing directory to the folder where the generated damage patterns are desired to be stored
main_directory = os.getcwd()
image_directory = main_directory + r"DamagePatterns\Raw\\"
os.chdir(image_directory)

# Dimensions for plate and hole in centimetres (whole major and minor axis, not half)
L = 10
W = 10
t_lam = 1
Sa = 2
Sb = 4

# No. of different non-zero loading conditions in X and Y direction
nx = 20
ny = 20

# Load variable initialisation - the actual load in centimetres is the load variable divided by 100
X = 0
Y = 0

# Loop to generate results from models until maximum load values in both directions are reached
while X <= 100:

    while Y <= 100:

        # The '%03d'% operator is used to convert the integer into a 3-digit string (25 turns into 025)
        # This makes sure that 25 comes before 100 when arranged in alphabetical order.
        modelName = 'Composite_L' + str(L) + '_W' + str(W) + '_t' + str(t_lam) + '_Sa' + str(Sa) + '_Sb' + str(Sb) \
                    + '_X' + '%03d'% X + '_Y' + '%03d'% Y

        model_directory = main_directory + modelName
        empty_image_directory = image_directory + 'EmptyFolder_ ' + modelName

        # Run code only if the damage patterns for the model do not already exist - the empty folder acts as a flag
        if not os.path.exists(empty_image_directory):

            # Open file
            odb_file = model_directory + r"\\" + modelName + ".odb"
            o1 = session.openOdb(name=odb_file)
            session.viewports['Viewport: 1'].setValues(displayedObject=o1)

            os.makedirs(empty_image_directory)

            # Contour Options
            session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
                renderStyle=FILLED, visibleEdges=FEATURE)
            session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
                intervalType=USER_DEFINED, intervalValues=(0, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1, 2, ))
            session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(numIntervals=11)

            # Viewport Annotation Options
            session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(triad=OFF,
                    legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)

            # Select HSNFCCRT
            session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
            session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
                variableLabel='HSNFCCRT', outputPosition=INTEGRATION_POINT, )

            # Print layers 1, 2, 3, 4
            for i in range(1,5):

                # Select layer i
                session.viewports['Viewport: 1'].odbDisplay.setPrimarySectionPoint(activePly="PLY-" + str(i))
                session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(sectionPointScheme=PLY_BASED)

                # Print layer i HSNFCCRT
                session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
                session.pngOptions.setValues(imageSize=(589, 279))
                session.printOptions.setValues(vpDecorations=OFF)
                session.printToFile(fileName='FC_Ply' + str(i) + '_' + modelName,
                    format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

            # Select HSNFTCRT
            session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
            session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
                    variableLabel='HSNFTCRT', outputPosition=INTEGRATION_POINT, )

            # Print layers 1, 2, 3, 4
            for i in range(1,5):

                # Select layer i
                session.viewports['Viewport: 1'].odbDisplay.setPrimarySectionPoint(activePly="PLY-" + str(i))
                session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(sectionPointScheme=PLY_BASED)

                # Print layer i HSNFTCRT
                session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
                session.pngOptions.setValues(imageSize=(589, 279))
                session.printOptions.setValues(vpDecorations=OFF)
                session.printToFile(fileName='FT_Ply' + str(i) + '_' + modelName,
                    format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

            # Select HSNMCCRT
            session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
            session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
                    variableLabel='HSNMCCRT', outputPosition=INTEGRATION_POINT, )

            # Print layers 1, 2, 3, 4
            for i in range(1, 5):

                # Select layer i
                session.viewports['Viewport: 1'].odbDisplay.setPrimarySectionPoint(activePly="PLY-" + str(i))
                session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(sectionPointScheme=PLY_BASED)

                # Print layer i HSNMCCRT
                session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
                session.pngOptions.setValues(imageSize=(589, 279))
                session.printOptions.setValues(vpDecorations=OFF)
                session.printToFile(fileName='MC_Ply' + str(i) + '_' + modelName,
                    format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

            # Select HSNMTCRT
            session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
            session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
                    variableLabel='HSNMTCRT', outputPosition=INTEGRATION_POINT, )

            # Print layers 1, 2, 3, 4
            for i in range(1, 5):

                # Select layer i
                session.viewports['Viewport: 1'].odbDisplay.setPrimarySectionPoint(activePly="PLY-" + str(i))
                session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(sectionPointScheme=PLY_BASED)

                # Print Layer i HSNMTCRT
                session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
                session.pngOptions.setValues(imageSize=(589, 279))
                session.printOptions.setValues(vpDecorations=OFF)
                session.printToFile(fileName='MT_Ply' + str(i) + '_' + modelName,
                    format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

        Y = Y + 100/ny

    Y = 0
    X = X + 100/nx