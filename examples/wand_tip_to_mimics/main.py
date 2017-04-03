# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 10:09:10 2017

@author: u0078867
"""

import sys
import os

modPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.append(modPath)


from PyBiomech import procedure_or as proc

# Necessary arguments
filePathC3D = 'FMCC.c3d'
filePathMimics = 'landmarks.mimics.txt'
wantTipName = 'MyPoint'
refSegment = 'femur'

# Optional arguments
filePathNewC3D = 'FMCC.c3d'
segSTLFilePath = 'femur.stl'
verbose = False
forceNoPauses = True

# Read Mimics file
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

print tip