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
from PyBiomech import vtkh, fio
import csv
import glob, os
import re
import numpy as np


# Necessary arguments
folderPath = './files/'
filePathMimics = './files/Global_marker_points.txt'
verbose = True
sideClusterAssoc = {
    'right': ['mF1','mF2','mF3','mF4'],
    'left': ['mT1','mT2','mT3','mT4'],
}
showNavigator = True
useWandMarkersFromFile = False
wandMarkersFile = './files/wand_markers.txt'

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

for filePath in filePaths:
    
    # Get path of current C3D file
    filePathC3D = filePath
    print('Processing file %s ...' % filePathC3D)
    
    # Remove extension from path
    filePathNoExt = os.path.splitext(filePath)[0]

    # Init data to show
    dataToShow = []
    
    # Parse side
    m = re.search(r'(right)|(left)', filePath, re.I)
    if m:
        side = m.group().lower()
        print('Parsed side: %s' % side)
    else:
        print('No parsed side, skipping processing')
        continue
    
    # Get reference segment to use
    refSegment = sideClusterAssoc[side]
    print('Reference segment: %s' % str(refSegment))
    
    # Read Mimics file
    tip, tipReduced = proc.expressOptoWandTipToMimicsRefFrame(
                                            filePathC3D, 
                                            filePathMimics, 
                                            'MyPoint', 
                                            refSegment,
                                            wandParams = wand,
                                            filePathNewC3D = None,
                                            reduceAs = None,
                                            segSTLFilePath = None,
                                            verbose = verbose,
                                            showNavigator = False,
                                            forceNoPauses = True
                                            )
    
     
    # Write point cloud coordinates to file
    filePathTXT = filePathNoExt + '.txt'
    with open(filePathTXT, 'wb') as f:
        fieldNames = ['x', 'y', 'z']
        writer = csv.DictWriter(f, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows([{'x':t[0], 'y':t[1], 'z':t[2]} for t in tip])
        
    # Add STL mesh to data to show
    item = {}
    item['name'] = 'clavicula'
    item['type'] = 'STL'
    if side == 'right':
        filePathSTL = os.path.join(folderPath, 'Rightclavicle.stl')
    elif side == 'left':
        filePathSTL = os.path.join(folderPath, 'Leftclavicle.stl')
    item['filePath'] = filePathSTL
    item['opacity'] = 0.5
    dataToShow.append(item)
    
    # Create point cloud to export
    pointCloud = []
    for i in xrange(tip.shape[0]):
        item = {}
        item['name'] = 'point_' + str(i)
        item['type'] = 'point'
        item['coords'] = tip[i,:]
        item['radius'] = 1.
        pointCloud.append(item)
    dataToShow.extend(pointCloud)
    
    # Export to XML
    filePathXML = filePathNoExt + '.xml'
    fio.writeXML3Matic(filePathXML, pointCloud)

    if showNavigator:
        # Show actors
        vtkh.showData(dataToShow)




    
    