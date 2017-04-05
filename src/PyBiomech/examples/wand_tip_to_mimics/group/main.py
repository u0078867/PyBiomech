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
from PyBiomech import vtkh
import glob, os
import numpy as np
import csv


# Set constant parameters
folderPath = './'   # folder path containing C3Ds
filePathMimics = 'Spec4_15L_GlobalCS_Landmarks.txt'
refSegment = 'tibia'
segSTLFilePath = 'Tibiaoriginal.stl'
verbose = False
forceNoPauses = True

# Search C3D files
filePaths = glob.glob(os.path.join(folderPath, '*.c3d'))
fileNames = [os.path.basename(fp) for fp in filePaths]
#print fileNames

# Group files by sensor, gage, gage tip
template = '%s_%s_%s_%s'
pattern = '(\w+)_(\w+)_(\w+)_(\w+)'
groups = proc.groupListBy(fileNames, pattern, lambda x: x[:3])
print(groups.keys())

tips = {}
for g in groups:
    
    tips[g] = np.zeros((0,3))
    
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
            tip = proc.expressOptoWandTipToMimicsRefFrame(
                                                    filePathC3D, 
                                                    filePathMimics, 
                                                    wantTipName, 
                                                    refSegment,
                                                    filePathNewC3D = filePathNewC3D,
                                                    segSTLFilePath = segSTLFilePath,
                                                    verbose = verbose,
                                                    forceNoPauses = forceNoPauses
                                                    )
                                                    
        except:
            tip = np.zeros((3,)) * np.nan
        
        # Insert tip in data matrix
        tips[g] = np.vstack([tips[g], tip])

# Restructure tips data for easier handling
tips = [{
    'name': t, 
    'coords': np.nanmean(tips[t], axis=0).tolist(),
    'std': np.nanstd(tips[t], axis=0).tolist()
} for t in tips]

# Write tip coordinates to file
with open('tips.txt', 'w') as f:
    fieldNames = ['name', 'coords', 'std']
    writer = csv.DictWriter(f, fieldnames=fieldNames)
    writer.writeheader()
    writer.writerows(tips)

# Create data structure for VTK navigator
data = []
for t in tips:
    item = {}
    item['name'] = t['name']
    item['type'] = 'point'
    item['coords'] = t['coords']
    item['radius'] = 1
    sensor = t['name'][0]
    if sensor == 'sensor1':
        item['color'] = (1,0,0)
    elif sensor == 'sensor2':
        item['color'] = (0,1,0)
    elif sensor == 'sensor3':
        item['color'] = (0,0,1)
    data.append(item)
item = {}
item['name'] = 'tibia'
item['type'] = 'STL'
item['filePath'] = segSTLFilePath
item['opacity'] = 0.5
data.append(item)

vtkh.showData(data)
    
    
    
    
    