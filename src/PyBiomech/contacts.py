# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 10:35:47 2017

@author: u0078867
"""

import vtk
from . import vtkh


  
    
def calculateBonesContactData(
                                vtkBoneA,
                                vtkBoneB,
                             ):

    distanceFilter = vtk.vtkDistancePolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        distanceFilter.SetInput(0, vtkBoneA)
        distanceFilter.SetInput(1, vtkBoneB)
    else:
        distanceFilter.SetInputData(0, vtkBoneA)
        distanceFilter.SetInputData(1, vtkBoneB)
    distanceFilter.Update()
    
    vtkBoneADistance = distanceFilter.GetOutput()
    vtkBoneBDistance = distanceFilter.GetSecondDistanceOutput()
    
    return vtkBoneADistance, vtkBoneBDistance



def calculateBonesContactAnalysisROIonScaledBB(vtkFemur, vtkTibia, scales1, scales2):

    vtkFemurBox = vtkh.getBoundingBox(vtkFemur)
    
    vtkFemurScaledBox = vtkh.scaleVTKDataAroundCenter(vtkFemurBox, scales1)
 
    vtkTibia2 = vtkh.clipVTKDataWithBox(vtkTibia, vtkFemurScaledBox.GetBounds())

    vtkTibiaBox = vtkh.getBoundingBox(vtkTibia)

    vtkTibiaScaledBox = vtkh.scaleVTKDataAroundCenter(vtkTibiaBox, scales2)

    vtkFemur2 = vtkh.clipVTKDataWithBox(vtkFemur, vtkTibiaScaledBox.GetBounds())
   
    return vtkFemur2, vtkTibia2

