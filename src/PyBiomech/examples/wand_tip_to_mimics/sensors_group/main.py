# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 10:09:10 2017

@author: u0078867
"""

import sys
import os

modPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../')
sys.path.append(modPath)


from PyBiomech import procedure_or as proc
from PyBiomech import vtkh, fio, kine, kine_or
import glob, os
import numpy as np
import csv


# Set constant parameters
folderPath = './'   # folder path containing C3Ds
filePathMimics = 'Spec4_15L_GlobalCS_Landmarks.txt'
refSegment = 'tibia'
segSTLFilePath = '4_15Lnewsegmentation.stl'
verbose = False
showNavigator = True
showSensorModelsOnly = True
forceNoPauses = True    # Set to True to process without interruption
useWandMarkersFromFile = True
wandMarkersFile = 'wand_markers.txt'



# Init data to show
dataToShow = []

# Add tibia mesh to data to show
item = {}
item['name'] = 'tibia'
item['type'] = 'STL'
item['filePath'] = segSTLFilePath
item['opacity'] = 0.5
dataToShow.append(item)

# Set custom wand params if needed
wand = None
if useWandMarkersFromFile:
    wand = {}
    wand['markers'] = ['WN', 'WW', 'WE', 'WM', 'WS']
    pointsMimics = fio.readMimics(wandMarkersFile,['markers'])['markers']
    wand['pos'] = {
        'WN': np.array(pointsMimics['Marker Top']),
        'WW': np.array(pointsMimics['Marker Left']),
        'WE': np.array(pointsMimics['Marker Right']),
        'WM': np.array(pointsMimics['Marker Mid']),
        'WS': np.array(pointsMimics['Marker Bottom']),
        'Tip': np.array(pointsMimics['Tip'])
    }
    wand['algoSVD'] = 2
    wand['algoTrilat'] = 0

# Search C3D files
filePaths = glob.glob(os.path.join(folderPath, '*.c3d'))
fileNames = [os.path.basename(fp) for fp in filePaths]

# Group files by sensor, gage, gage tip
template = '%s_%s_%s_%s'
pattern = '(\w+)_(\w+)_(\w+)_(\w+)'
groups = proc.groupListBy(fileNames, pattern, lambda x: x[:3])

# Loop each group
tips = {}
for g in groups:
    
    tips[g] = np.zeros((0,3))
    
    # Loop each file of the group
    for tokens in groups[g]:
    
        fileName = template % tokens
        
        # Necessary arguments
        filePathC3D = os.path.join(folderPath, fileName + '.c3d')
        print(filePathC3D)
        
        wantTipName = '_'.join(tokens[:3])
        
        # Optional arguments
        #filePathNewC3D = os.path.join(folderPath, fileName + '_Tip.c3d')
        filePathNewC3D = None
        
        # Read Mimics file
        try:
            dummy, tip = proc.expressOptoWandTipToMimicsRefFrame(
                                                    filePathC3D, 
                                                    filePathMimics, 
                                                    wantTipName, 
                                                    refSegment,
                                                    wandParams = wand,
                                                    filePathNewC3D = filePathNewC3D,
                                                    segSTLFilePath = segSTLFilePath,
                                                    reduceAs = 'avg_point',
                                                    verbose = verbose,
                                                    showNavigator = showNavigator,
                                                    forceNoPauses = forceNoPauses
                                                    )
                                                    
        except:
            tip = np.zeros((3,)) * np.nan
        
        # Insert tip in data matrix
        tips[g] = np.vstack([tips[g], tip])

# Restructure tips data for easier handling
tips2 = [{
    'name': '_'.join(t[:3]), 
    'coords': np.nanmean(tips[t], axis=0).tolist(),
    'std': np.nanstd(tips[t], axis=0).tolist()
} for t in tips]

# Write tip coordinates to file
with open('tips.txt', 'wb') as f:
    fieldNames = ['name', 'coords', 'std']
    writer = csv.DictWriter(f, fieldnames=fieldNames)
    writer.writeheader()
    writer.writerows(tips2)

# Create wand tip output
wandTipdata = []
for t in tips2:
    item = {}
    item['name'] = t['name']
    item['type'] = 'point'
    item['coords'] = t['coords']
    item['radius'] = 1
    sensor = item['name'][:7]
    if sensor == 'sensor1':
        item['color'] = (1,0,0)
    elif sensor == 'sensor2':
        item['color'] = (0,1,0)
    elif sensor == 'sensor3':
        item['color'] = (0,0,1)
    wandTipdata.append(item)

if not showSensorModelsOnly:
    # Add wand tip data to data to show
    dataToShow.extend(wandTipdata)

# Export to 3-matic
fio.writeXML3Matic('tips.xml', wandTipdata)

# Get sensor planar model
model = kine_or.get3StrainGagesSensorModel()

# Iterate over the 3 sensors
sensorModelsData = []
for i in xrange(1, 4):
    
    print('Reposing model for sensor %d ...' % i)    
    
    # Calculate sensor model pose
    prefix = 'sensor' + str(i) + '_'
    svdArgs = {}
    svdArgs['mkrsLoc'] = {(prefix + p): model['pos'][p] for p in model['pos']}
    sensorPointNames = [(prefix + m) for m in model['pos'].keys()]
    sensorPoints = {t['name']: np.array(t['coords'])[None,:] for t in tips2 if t['name'] in sensorPointNames}
    R, T = kine.rigidBodySVDFun(sensorPoints, sensorPointNames, svdArgs)
    RT = kine.composeRotoTranslMatrix(R, T)
    
    # Express sensor model points in Mimics reference frame
    sensorModelPoints = kine.changeMarkersReferenceFrame(svdArgs['mkrsLoc'], RT)
    sensorModelPoints = {p + '_model': sensorModelPoints[p].squeeze() for p in sensorModelPoints}
    
    # Create sensor models output
    for p in sensorModelPoints:
    
        item = {}
        item['name'] = p
        item['type'] = 'point'
        item['coords'] = sensorModelPoints[p]
        item['radius'] = 1
        sensor = item['name'][:7]
        if sensor == 'sensor1':
            item['color'] = (1,0,0)
        elif sensor == 'sensor2':
            item['color'] = (0,1,0)
        elif sensor == 'sensor3':
            item['color'] = (0,0,1)
        sensorModelsData.append(item)
        
# Export to 3-matic
fio.writeXML3Matic('sensor_model.xml', sensorModelsData)

if showSensorModelsOnly:
    # Add reposed sensor models data to data to show
    dataToShow.extend(sensorModelsData)

# Show actors
vtkh.showData(dataToShow)
    
    
    
    
    