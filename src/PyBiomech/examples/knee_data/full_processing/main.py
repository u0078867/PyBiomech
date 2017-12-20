# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:39:01 2017

@author: u0078867
"""


import sys
import os

modPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../')
sys.path.append(modPath)


from PyBiomech import procedure_or as proc
from PyBiomech import fio




# Set arguments
filePathC3D = './Native - PassiveFlexion Trial 1.c3d'
filePathORParams = './params.json'
filePathSTLFemur = './femur.stl'
filePathSTLTibia = './tibia.stl'
folderContactSceneFiles = './contact_scenes'
folderLigamentsSceneFiles = './ligaments_scenes'
assembleDataMethod = 2

# Read params JSON file and adjust a few structures for later
params = fio.readORParamsFile(filePathORParams)
markersLoc = {m: params['mkrsLoc'][m]['pos'] for m in params['mkrsLoc']}
linesLoc = {m: params['splines'][m]['pos'] for m in params['splines']}

# Read C3D file
print('==== Reading C3D file ...')
markers = fio.readC3D(filePathC3D, ['markers'], {
    'setMarkersZeroValuesToNaN': True,
    'removeSegmentNameFromMarkerNames': True,
})['markers']

# Calculate knee segments poses

poses = proc.calculateKneeSegmentsPoses(
                                        markers,
                                        markersLoc,
                                        verbose = False
                                       )
                                      
# Calculate knee kinematics data
RT1 = poses['femur_pose']
RT2 = poses['tibia_pose']
side = params['side']
kine = proc.calculateKneeKinematics(
                                    RT1,
                                    RT2,
                                    markersLoc,
                                    side
                                   )
                                   
# Calculate knee ligaments data
tibiaPlateauMedEdgeLine = linesLoc['TibiaPlateauMedEdge']
vtkFemur = fio.readSTL(filePathSTLFemur)
vtkTibia = fio.readSTL(filePathSTLTibia)
liga = proc.calculateKneeLigamentsData(
                                        RT1,
                                        RT2,
                                        markersLoc,
                                        frames = [300],
                                        ligaNames = ['MCL'],
                                        tibiaPlateauMedEdgeSplineLoc = tibiaPlateauMedEdgeLine,
#                                        ligaModels = ['straight','Blankevoort_1991', 'Marai_2004'],
                                        ligaModels = ['straight','Blankevoort_1991'],
                                        vtkFemur = vtkFemur,
                                        vtkTibia = vtkTibia,
                                        Marai2004Params = {'Ns': 10},
                                        saveScene = True,
                                        sceneFormats = ['vtm'],
                                        outputDirSceneFile = folderLigamentsSceneFiles,
                                        )

# Calculate knee segments contact data                         
contact = proc.calculateKneeContactsData(
                                            RT1,
                                            RT2,
                                            vtkFemur,
                                            vtkTibia,
                                            frames = [300],
                                            femurDecimation = 0.4,
                                            tibiaDecimation = 0.4,
                                            saveScene = True,
                                            sceneFormats = ['vtm'],
                                            outputDirSceneFile = folderContactSceneFiles,
                                        )
                                        
# Assemble all knee data
if assembleDataMethod == 1:
    kneeData = proc.assembleKneeDataGranular(
                                                markers=markers, 
                                                poses=poses, 
                                                kine=kine, 
                                                liga=liga, 
                                                contact=contact
                                            )
elif assembleDataMethod == 2:
    kneeData = proc.assembleKneeDataAsIsNoMetadata(
                                                    markers=markers, 
                                                    poses=poses, 
                                                    kine=kine, 
                                                    liga=liga, 
                                                    contact=contact
                                                  )

                                        
# Save data for file
print('==== Saving knee data ...')
fio.writeMATFile('knee_data.mat', kneeData)


print('==== Finished!')







