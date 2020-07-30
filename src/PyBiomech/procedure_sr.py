"""
.. module:: procedure_sr
   :synopsis: helper module for procedures used with Shoulder-Rig (IORT UZLeuven)

"""

import numpy as np

from . import fio, mplh, kine, vtkh, ligaments as liga, contacts, spine

import re
from itertools import groupby

import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



            
            


def calculateShoulderSegmentsPoses(
                            markersOpto,
                            markersLoc,
                            verbose = False,
                            vtkScapula = None,
                            vtkHumerus = None,
                            saveScene = False,
                            sceneFrames = None,
                            sceneFormats = ['vtm'],
                            outputDirSceneFile = None,
                            ):
    
    # Calculate pose from Mimics reference frame to Optoelectronic reference frame
    print('==== Calculating poses from Mimics reference frame to Optoelectronic reference frame ...')

    args = {}
    args['mkrsLoc'] = markersLoc
    args['verbose'] = verbose
    
    scapulaAvailable = False
    scapulaError = False
    mkrList = ["mS1", "mS2", "mS3", "mS4"]
    if set(mkrList) <= set(markersLoc.keys()):
        scapulaAvailable = True
        print('==== ---- Scapula markers found')
        try:
            R1, T1, info1 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
            RT1 = kine.composeRotoTranslMatrix(R1, T1)
        except Exception as e:
            print('Impossible to calculate scapula pose')
            print(e)
            scapulaError = True
    
    humerusAvailable = False
    humerusError = False
    mkrList = ["mH1", "mH2", "mH3", "mH4"]
    if set(mkrList) <= set(markersLoc.keys()):
        humerusAvailable = True
        print('==== ---- Humerus markers found')
        try:
            R2, T2, info2 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
            RT2 = kine.composeRotoTranslMatrix(R2, T2)
        except Exception as e:
            print('Impossible to calculate humerus pose')
            print(e)
            humerusError = True
    
    results = {}
    if scapulaAvailable and not scapulaError:
        results['scapula_pose'] = RT1
        results['scapula_pose_reconstr_info'] = info1
    if humerusAvailable and not humerusError:
        results['humerus_pose'] = RT2
        results['humerus_pose_reconstr_info'] = info2
    
    Nf = RT1.shape[0]

    iRange = range(Nf)
    if sceneFrames is not None:
        iRange = sceneFrames
        
    if saveScene and vtkScapula is not None and not scapulaAvailable:
        
        raise Exception('"saveScene = True" option requires scapula data to be available if input "vtkScapula" is provided')
    
    if saveScene and vtkHumerus is not None and not humerusAvailable:
        
        raise Exception('"saveScene = True" option requires humerus data to be available if input "vtkHumerus" is provided')
    
    for i in iRange:
        
        if saveScene:
        
            print('==== ---- Saving scene for time frame %d' % (i))
        
            if vtkScapula is not None:
            
                vtkScapula2 = vtkh.reposeVTKData(vtkScapula, RT1[i,...])
                
            if vtkHumerus is not None:
                
                vtkHumerus2 = vtkh.reposeVTKData(vtkHumerus, RT2[i,...])
        
                
            actors = []
            names = []
            
            if vtkScapula is not None:
            
                actors.append(vtkh.createVTKActor(vtkScapula2))
                names.append('scapula')
                
            if vtkHumerus is not None:
            
                actors.append(vtkh.createVTKActor(vtkHumerus2))
                names.append('humerus')
            
            scene = vtkh.createScene(actors)
            
            for fmt in sceneFormats:
                vtkh.exportScene(scene, outputDirSceneFile + ('/poses_tf_%05d' % i), ext=fmt, names=names)
                
    
    return results
    

def calculateShoulderKinematics(
                            RT1,
                            RT2,
                            landmarksLoc,
                            side
                            ):
                                
    allALs = {}

    # express SCAPULA anatomical landmarks in lab reference frame
    print('==== Expressing scapula anatomical landmarks in Optoelectronic reference frame ...')
    ALs = {m: np.array(landmarksLoc[m]) for m in ["AA","IA","TS"]}
    allALs.update(kine.changeMarkersReferenceFrame(ALs, RT1))

    # calculate SCAPULA anatomical reference frame
    Ra1, Oa1 = kine.scapulaPose(allALs, s=side)

    # express HUMERUS anatomical landmarks in lab reference frame
    print('==== Expressing humerus anatomical landmarks in Optoelectronic reference frame ...')
    ALs = {m: np.array(landmarksLoc[m]) for m in ["GH","LE","ME"]}
    allALs.update(kine.changeMarkersReferenceFrame(ALs, RT2))

    # calculate HUMERUS anatomical reference frame
    Ra2, Oa2 = kine.humerusPose(allALs, s=side)

    # calculate SHOULDER kinematics
    print('==== Calculating shoulder kinematics ...')
    angles = kine.getJointAngles(Ra1, Ra2, R2anglesFun=kine.R2yxy, funInput='jointR')
    transl = kine.getJointTransl(Ra1, Ra2, Oa1, Oa2, T2translFun=kine.gesTranslFromSegment1)
    results = {}
    results["elevation_plane"] = angles[:,0]
    results["elevation"] = angles[:,1]
    results["intrarotation"] = angles[:,2]
    results["AP"] = transl[:,0]
    results["IS"] = transl[:,1]
    results["ML"] = transl[:,2]
    results["landmarks"] = allALs
    
    return results

    

def assembleShoulderDataAsIsNoMetadata(
                                    markersLoc=None,
                                    markers=None, 
                                    poses=None, 
                                    kine=None, 
                                  ):
                                      
    print('==== Assemblying shoulder data ...')
    
    data = {}
    
    if markersLoc is not None:
        
        data['markersLoc'] = markersLoc
    
    if markers is not None:
    
        data['markers'] = markers
        
        if kine is not None:
            
            data['markers'].update(kine['landmarks'])
            
    if poses is not None:
    
        data['poses'] = poses
        
    if kine is not None:
    
        data['kine'] = kine
    
    return data
    