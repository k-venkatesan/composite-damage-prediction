# DO NOT RUN THIS FILE IN YOUR PYTHON IDE - RUN IT ON ABAQUS

from abaqus import *
from abaqusConstants import *
import __main__
import os

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

# Clear runfile (refer to the README file of this folder for more details) - remember to set the directory correctly if
# the compiler cannot find 'runfile.txt'
main_directory = os.getcwd()
runfile = open("runfile.txt", "w+")
runfile.write('')
runfile.close()

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

# Loops to fill up runfile with model information until maximum load values in both directions are reached
while X <= 100:

    while Y <= 100:

        # The '%03d'% operator is used to convert the integer into a 3-digit string (25 turns into 025)
        # This makes sure that 25 comes before 100 when arranged in alphabetical order.
        model_name = 'Composite_L' + str(L) + '_W' + str(W) + '_t' + str(t_lam) + '_Sa' + str(Sa) + '_Sb' + str(Sb) \
                    + '_X' + '%03d'% X + '_Y' + '%03d'% Y

        # New model in new directory
        new_directory = main_directory + r'\abaqus-models\\' + model_name

        # Run this iteration of the loop only if the model does not already exist
        if not os.path.exists(new_directory):

            os.makedirs(new_directory)
            os.chdir(new_directory)

            # Create model
            mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)

            # Setup sketch
            s1 = mdb.models[model_name].ConstrainedSketch(name='__profile__', sheetSize=20.0)
            g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
            s1.setPrimaryObject(option=STANDALONE)

            # Sketch plate with elliptical hole
            s1.Line(point1=(-L/2, W/2), point2=(L/2, W/2))
            s1.HorizontalConstraint(entity=g[2], addUndoState=False)
            s1.Line(point1=(L/2, W/2), point2=(L/2, -W/2))
            s1.VerticalConstraint(entity=g[3], addUndoState=False)
            s1.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
            s1.Line(point1=(L/2, -W/2), point2=(-L/2, -W/2))
            s1.HorizontalConstraint(entity=g[4], addUndoState=False)
            s1.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
            s1.Line(point1=(-L/2, -W/2), point2=(-L/2, W/2))
            s1.VerticalConstraint(entity=g[5], addUndoState=False)
            s1.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
            s1.EllipseByCenterPerimeter(center=(0.0, 0.0), axisPoint1=(Sa/2, 0.0),
                axisPoint2=(0.0, Sb/2))

            # Create part
            p = mdb.models[model_name].Part(name='CompositePlate', dimensionality=THREE_D, type=DEFORMABLE_BODY)
            p = mdb.models[model_name].parts['CompositePlate']
            p.BaseShell(sketch=s1)
            s1.unsetPrimaryObject()
            del mdb.models[model_name].sketches['__profile__']

            # Create partitions
            f, e, d1 = p.faces, p.edges, p.datums
            t = p.MakeSketchTransform(sketchPlane=f[0], sketchUpEdge=e[3], sketchPlaneSide=SIDE1, origin=(0.0, 0.0, 0.0))
            s = mdb.models[model_name].ConstrainedSketch(name='__profile__', sheetSize=28.28, gridSpacing=0.7, transform=t)
            g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
            s.setPrimaryObject(option=SUPERIMPOSE)
            p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
            s.EllipseByCenterPerimeter(center=(0.0, 0.0), axisPoint1=(Sa, 0.0), axisPoint2=(0.0, Sb))
            s.Line(point1=(-Sa, 0.0), point2=(-L/2, 0.0))
            s.HorizontalConstraint(entity=g[9], addUndoState=False)
            s.PerpendicularConstraint(entity1=g[7], entity2=g[9], addUndoState=False)
            s.CoincidentConstraint(entity1=v[8], entity2=g[7], addUndoState=False)
            s.CoincidentConstraint(entity1=v[9], entity2=g[3], addUndoState=False)
            s.EqualDistanceConstraint(entity1=v[2], entity2=v[3], midpoint=v[9], addUndoState=False)
            s.Line(point1=(-Sa, 0.0), point2=(-Sa/2, 0.0))
            s.HorizontalConstraint(entity=g[10], addUndoState=False)
            s.ParallelConstraint(entity1=g[9], entity2=g[10], addUndoState=False)
            s.CoincidentConstraint(entity1=v[10], entity2=g[2], addUndoState=False)
            s.Line(point1=(0.0, Sb), point2=(0.0, W/2))
            s.VerticalConstraint(entity=g[11], addUndoState=False)
            s.PerpendicularConstraint(entity1=g[6], entity2=g[11], addUndoState=False)
            s.CoincidentConstraint(entity1=v[11], entity2=g[6], addUndoState=False)
            s.EqualDistanceConstraint(entity1=v[5], entity2=v[2], midpoint=v[11], addUndoState=False)
            s.CoincidentConstraint(entity1=v[12], entity2=g[7], addUndoState=False)
            s.Line(point1=(0.0, Sb), point2=(0.0, Sb/2))
            s.VerticalConstraint(entity=g[12], addUndoState=False)
            s.ParallelConstraint(entity1=g[11], entity2=g[12], addUndoState=False)
            s.CoincidentConstraint(entity1=v[13], entity2=g[2], addUndoState=False)
            s.Line(point1=(Sa, 0.0), point2=(L/2, 0.0))
            s.HorizontalConstraint(entity=g[13], addUndoState=False)
            s.PerpendicularConstraint(entity1=g[5], entity2=g[13], addUndoState=False)
            s.CoincidentConstraint(entity1=v[14], entity2=g[5], addUndoState=False)
            s.EqualDistanceConstraint(entity1=v[4], entity2=v[5], midpoint=v[14], addUndoState=False)
            s.Line(point1=(Sa, 0.0), point2=(Sa/2, 0.0))
            s.HorizontalConstraint(entity=g[14], addUndoState=False)
            s.ParallelConstraint(entity1=g[13], entity2=g[14], addUndoState=False)
            s.Line(point1=(0.0, -Sb), point2=(0.0, -W/2))
            s.VerticalConstraint(entity=g[15], addUndoState=False)
            s.PerpendicularConstraint(entity1=g[4], entity2=g[15], addUndoState=False)
            s.CoincidentConstraint(entity1=v[15], entity2=g[4], addUndoState=False)
            s.EqualDistanceConstraint(entity1=v[3], entity2=v[4], midpoint=v[15], addUndoState=False)
            s.CoincidentConstraint(entity1=v[16], entity2=g[7], addUndoState=False)
            s.Line(point1=(0.0, -Sb), point2=(0.0, -Sb/2))
            s.VerticalConstraint(entity=g[16], addUndoState=False)
            s.ParallelConstraint(entity1=g[15], entity2=g[16], addUndoState=False)
            s.CoincidentConstraint(entity1=v[17], entity2=g[2], addUndoState=False)
            pickedFaces = f.getSequenceFromMask(mask=('[#1 ]', ), )
            e1, d2 = p.edges, p.datums
            p.PartitionFaceBySketch(sketchUpEdge=e1[3], faces=pickedFaces, sketch=s)
            s.unsetPrimaryObject()
            del mdb.models[model_name].sketches['__profile__']

            # Seed edges
            e = p.edges
            pickedEdges = e.getSequenceFromMask(mask=('[#f20300 ]',), )
            p.seedEdgeByNumber(edges=pickedEdges, number=21, constraint=FINER)
            pickedEdges = e.getSequenceFromMask(mask=('[#80080 ]',), )
            p.seedEdgeByNumber(edges=pickedEdges, number=18, constraint=FINER)
            pickedEdges = e.getSequenceFromMask(mask=('[#10400 ]',), )
            p.seedEdgeByNumber(edges=pickedEdges, number=9, constraint=FINER)
            pickedEdges1 = e.getSequenceFromMask(mask=('[#1040 ]',), )
            pickedEdges2 = e.getSequenceFromMask(mask=('[#4008 ]',), )
            p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1,
                             end2Edges=pickedEdges2, ratio=4.0, number=28, constraint=FINER)
            pickedEdges2 = e.getSequenceFromMask(mask=('[#14 ]',), )
            p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=pickedEdges2, ratio=3.0, number=12, constraint=FINER)

            # Meshing
            f = p.faces
            pickedRegions = f.getSequenceFromMask(mask=('[#ff ]',), )
            p.setMeshControls(regions=pickedRegions, elemShape=TRI)
            elemType1 = mesh.ElemType(elemCode=S8R, elemLibrary=STANDARD)
            elemType2 = mesh.ElemType(elemCode=STRI65, elemLibrary=STANDARD)
            faces = f.getSequenceFromMask(mask=('[#ff ]',), )
            pickedRegions = (faces,)
            p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
            p.generateMesh()

            # Define material
            mdb.models[model_name].Material(name='CFRP')
            mdb.models[model_name].materials['CFRP'].Elastic(type=LAMINA, table=((161000.0, 11000.0, 0.32,
                                                                                 5170.0, 5170.0, 5170.0),))
            mdb.models[model_name].materials['CFRP'].elastic.FailStress(table=((2800.0, 1700.0, 60.0, 125.0,
                                                                                      90.0, 0.0, 0.0),))
            mdb.models[model_name].materials['CFRP'].HashinDamageInitiation(table=((2800.0, 1700.0, 60.0,
                                                                                          125.0, 90.0, 90.0),))
            mdb.models[model_name].materials['CFRP'].hashinDamageInitiation.DamageEvolution(type=ENERGY,
                                                                                           table=((100.0, 100.0, 0.22, 0.72),))
            mdb.models[model_name].materials['CFRP'].hashinDamageInitiation.DamageStabilization(
                fiberTensileCoeff=0.005, fiberCompressiveCoeff=0.005,
                matrixTensileCoeff=0.005, matrixCompressiveCoeff=0.005)

            # Create composite layup
            layupOrientation = None
            faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
            region1=regionToolset.Region(faces=faces)
            faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
            region2=regionToolset.Region(faces=faces)
            faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
            region3=regionToolset.Region(faces=faces)
            faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
            region4=regionToolset.Region(faces=faces)
            compositeLayup = mdb.models[model_name].parts['CompositePlate'].CompositeLayup(
                name='CFRPLayup', description='', elementType=SHELL,
                offsetType=MIDDLE_SURFACE, symmetric=True,
                thicknessAssignment=FROM_SECTION)
            compositeLayup.Section(preIntegrate=OFF, integrationRule=SIMPSON,
                thicknessType=UNIFORM, poissonDefinition=DEFAULT, temperature=GRADIENT,
                useDensity=OFF)
            compositeLayup.ReferenceOrientation(orientationType=GLOBAL, localCsys=None,
                fieldName='', additionalRotationType=ROTATION_NONE, angle=0.0,
                axis=AXIS_3)
            compositeLayup.CompositePly(suppressed=False, plyName='Ply-1',
                region=region1, material='CFRP', thicknessType=SPECIFY_THICKNESS,
                thickness=0.125, orientationType=SPECIFY_ORIENT, orientationValue=45.0,
                additionalRotationType=ROTATION_NONE, additionalRotationField='',
                axis=AXIS_3, angle=0.0, numIntPoints=1)
            compositeLayup.CompositePly(suppressed=False, plyName='Ply-2', region=region2,
                material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=0.125,
                orientationType=SPECIFY_ORIENT, orientationValue=90.0,
                additionalRotationType=ROTATION_NONE, additionalRotationField='',
                axis=AXIS_3, angle=0.0, numIntPoints=1)
            compositeLayup.CompositePly(suppressed=False, plyName='Ply-3', region=region3,
                material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=0.125,
                orientationType=SPECIFY_ORIENT, orientationValue=-45.0,
                additionalRotationType=ROTATION_NONE, additionalRotationField='',
                axis=AXIS_3, angle=0.0, numIntPoints=1)
            compositeLayup.CompositePly(suppressed=False, plyName='Ply-4', region=region4,
                material='CFRP', thicknessType=SPECIFY_THICKNESS, thickness=0.125,
                orientationType=SPECIFY_ORIENT, orientationValue=0.0,
                additionalRotationType=ROTATION_NONE, additionalRotationField='',
                axis=AXIS_3, angle=0.0, numIntPoints=1)

            # Assembly
            a = mdb.models[model_name].rootAssembly
            a.DatumCsysByDefault(CARTESIAN)
            a.Instance(name='CompositePlate-1', part=p, dependent=ON)

            # Step
            mdb.models[model_name].StaticStep(name='Load', previous='Initial',
                                                    maxNumInc=1000, initialInc=1e-05, minInc=1e-20, nlgeom=ON)

            # Field output requests
            mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(
                variables=('HSNFTCRT', 'HSNFCCRT', 'HSNMTCRT', 'HSNMCCRT'),
                layupNames=('CompositePlate-1.CFRPLayup',),
                layupLocationMethod=SPECIFIED, outputAtPlyTop=False,
                outputAtPlyMid=True, outputAtPlyBottom=False, rebar=EXCLUDE)

            # Boundary conditions
            e1 = a.instances['CompositePlate-1'].edges
            edges1 = e1.getSequenceFromMask(mask=('[#820000 ]', ), )
            region = regionToolset.Region(edges=edges1)
            mdb.models[model_name].DisplacementBC(name='Bottom',
                createStepName='Load', region=region, u1=UNSET, u2=-Y*0.001, u3=UNSET,
                ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF,
                distributionType=UNIFORM, fieldName='', localCsys=None)
            e1 = a.instances['CompositePlate-1'].edges
            edges1 = e1.getSequenceFromMask(mask=('[#600000 ]', ), )
            region = regionToolset.Region(edges=edges1)
            mdb.models[model_name].DisplacementBC(name='Left', createStepName='Load',
                region=region, u1=-X*0.001, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET,
                ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM,
                fieldName='', localCsys=None)
            e1 = a.instances['CompositePlate-1'].edges
            edges1 = e1.getSequenceFromMask(mask=('[#100200 ]', ), )
            region = regionToolset.Region(edges=edges1)
            mdb.models[model_name].DisplacementBC(name='Top', createStepName='Load',
                region=region, u1=UNSET, u2=Y*0.001, u3=UNSET, ur1=UNSET, ur2=UNSET,
                ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM,
                fieldName='', localCsys=None)
            e1 = a.instances['CompositePlate-1'].edges
            edges1 = e1.getSequenceFromMask(mask=('[#40100 ]', ), )
            region = regionToolset.Region(edges=edges1)
            mdb.models[model_name].DisplacementBC(name='Right',
                createStepName='Load', region=region, u1=X*0.001, u2=UNSET, u3=UNSET,
                ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF,
                distributionType=UNIFORM, fieldName='', localCsys=None)

            # Create job
            mdb.Job(name=model_name, model=model_name, description='',
                    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None,
                    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
                    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1,
                    numGPUs=0)
            mdb.jobs[model_name].writeInput(consistencyChecking=OFF)

            # Append run file

            os.chdir(main_directory)
            runfile = open("runfile.txt", "a+")
            runfile.write('\ncd ' + new_directory + '\n')
            runfile.write(r'call c:\simulia\commands\abaqus.bat j=' + model_name + ' -seq \n')
            runfile.close()

        Y = Y + 100/ny

    Y = 0
    X = X + 100/nx