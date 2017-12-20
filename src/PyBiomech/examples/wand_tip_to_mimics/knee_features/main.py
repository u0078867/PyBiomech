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
from PyBiomech import vtkh, fio, kine
from PyBiomech import mplh
import csv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
import glob, os
import re



# Necessary arguments
folderPath = './files/'
filePathMimics = './files/Spec4_15R_GlobalCS_Landmarks.txt'
segSTLFilePath = './files/04-15-Right.stl'
verbose = False
showNavigator = True

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
    
    # Parse reference segment
    m = re.search(r'(tibia)|(femur)', filePath, re.I)
    if m:
        refSegment = m.group().lower()
        print('Parsed reference segment: %s' % refSegment)
    else:
        print('No parsed reference segment, skipping processing')
        continue

    # Init data to show
    dataToShow = []
    
    # Add tibia mesh to data to show
    item = {}
    item['name'] = 'tibia'
    item['type'] = 'STL'
    item['filePath'] = segSTLFilePath
    item['opacity'] = 1.
    dataToShow.append(item)
    
    # Read Mimics file
    tip, tipReduced = proc.expressOptoWandTipToMimicsRefFrame(
                                            filePathC3D, 
                                            filePathMimics, 
                                            'MyPoint', 
                                            refSegment,
                                            filePathNewC3D = None,
                                            reduceAs = None,
                                            segSTLFilePath = segSTLFilePath,
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
    
    # Create line item to show
    item = {}
    item['name'] = 'tracked_line'
    item['type'] = 'line'
    item['coords'] = tip
    item['color'] = (255,0,0)
    dataToShow.append(item)
    
    # Create point cloud to export
    pointCloud = []
    for i in xrange(tip.shape[0]):
        item = {}
        item['name'] = 'point_' + str(i)
        item['type'] = 'point'
        item['coords'] = tip[i,:]
        pointCloud.append(item)
    
    # Export to XML
    filePathXML = filePathNoExt + '.xml'
    fio.writeXML3Matic(filePathXML, pointCloud)
    
    # Parse reference segment
    m = re.search(r'(ridge)|(surface)', filePath, re.I)
    scannedGeometry = None
    if m:
        scannedGeometry = m.group().lower()
        print('Parsed scanned geometry: %s' % scannedGeometry)
    else:
        print('No parsed scanned geometry')
    
    if scannedGeometry == 'ridge':
    
        # Fit stright line with SVD
        versor, center, other = kine.lineFitSVD(tip)
        proj = other['proj']
        linePoints = versor * np.mgrid[proj.min():proj.max():30j][:, np.newaxis]
        linePoints += center
        
        # Create straight line to export
        item = {}
        item['name'] = 'fitted_straight_line'
        item['type'] = 'line'
        item['coords'] = linePoints
        item['color'] = (0,255,0)
        dataToShow.append(item)
        
        # Export to XML
        filePathXML = filePathNoExt + '_straight_line.xml'
        fio.writeXML3Matic(filePathXML, [item])
        
        # Plot 3D regression
        ax = m3d.Axes3D(plt.figure())
        ax.plot3D(tip[:,0], tip[:,1], tip[:,2], 'bo', label='original')
        ax.plot3D(linePoints[:,0], linePoints[:,1], linePoints[:,2], 'ro', label='projected')
        ax.set_aspect('equal')
        mplh.set_axes_equal(ax)
        plt.legend()
        plt.show()
        
    elif scannedGeometry == 'surface':
        
        if refSegment == 'tibia':
            
            # Fit plane with SVD
            versor, center, other = kine.planeFitSVD(tip)
            planePoints = other['proj']
            
            # Create line item to show
            item = {}
            item['name'] = 'tracked_line_projected'
            item['type'] = 'line'
            item['coords'] = planePoints
            item['color'] = (0,255,0)
            dataToShow.append(item)
            
            # Create straight to export
            pointCloud = []
            for i in xrange(planePoints.shape[0]):
                item = {}
                item['name'] = 'point_' + str(i) + '_projected'
                item['type'] = 'point'
                item['coords'] = planePoints[i,:]
                pointCloud.append(item)
            
            # Export to XML
            filePathXML = filePathNoExt + '_proj_on_plane.xml'
            fio.writeXML3Matic(filePathXML, pointCloud)
            
            # Plot 3D regression
            ax = m3d.Axes3D(plt.figure())
            ax.plot3D(tip[:,0], tip[:,1], tip[:,2], 'bo', label='original')
            ax.plot3D(planePoints[:,0], planePoints[:,1], planePoints[:,2], 'ro', label='projected')
            ax.set_aspect('equal')
            mplh.set_axes_equal(ax)
            plt.legend()
            plt.show()
    
    if showNavigator:
        # Show actors
        vtkh.showData(dataToShow)
    
    
    
    
        
        