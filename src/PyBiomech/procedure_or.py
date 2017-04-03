"""
.. module:: procedure_or
   :synopsis: helper module for procedures used with Oxford-Rig (IORT UZLeuven)

"""

import numpy as np
import fio, kine, kine_or, vtkh


def expressOptoWandTipToMimicsRefFrame(
                                        filePathC3D, 
                                        filePathMimics, 
                                        wantTipName, 
                                        refSegment,
                                        filePathNewC3D = None,
                                        segSTLFilePath = None,
                                        verbose = False,
                                        forceNoPauses = False
                                        ):
    # Read Mimics file
    print('==== Reading MIMICS file ...')
    markersMimics = fio.readMimics(filePathMimics, ['markers'])['markers']
    
    # Read C3D
    print('==== Reading C3D file ...')
    markersOpto = fio.readC3D(filePathC3D, ['markers'], {
        'removeSegmentNameFromMarkerNames': True
    })['markers']
    
    # Calculate wand tip in Optoelectronic reference frame
    print('==== Reconstructing wand tip ...')
    tipOpto = kine_or.getPointerTipOR(markersOpto, verbose=verbose)
    
    # Write to C3D
    if filePathNewC3D is not None:
        print('==== Writing C3D file with tip ...')
        data = {}
        data['markers'] = {}
        data['markers']['data'] = markersOpto
        data['markers']['data'][wantTipName] = tipOpto
        fio.writeC3D(filePathNewC3D, data, copyFromFile=filePathC3D)
    
    # Calculate pose from Mimics reference frame to Optoelectronic reference frame
    print('==== Calculating {0} pose from Mimics reference frame to Optoelectronic reference frame ...'.format(refSegment))
    if refSegment == 'femur':
        mkrList = ['mF1', 'mF2', 'mF3', 'mF4']
    elif refSegment == 'tibia':
        mkrList = ['mT1', 'mT2', 'mT3', 'mT4']
    args = {}
    args['mkrsLoc'] = markersMimics
    args['verbose'] = verbose
    oRm, oTm = kine.rigidBodySVDFun(markersOpto, mkrList, args)
    oRTm = kine.composeRotoTranslMatrix(oRm, oTm)
    
    # Calculate pose from Optoelectronic reference frame to Mimics reference frame
    print('==== Inveting pose ...')
    mRTo = kine.inv2(oRTm)
    
    # Express tip in Mimics reference frame
    print('==== Expressing tip in Mimics reference frame ...')
    tipMimics = kine.changeMarkersReferenceFrame({'tip': tipOpto}, mRTo)['tip']
    tipMimicsAvg = np.mean(tipMimics, axis=0)
    
    if segSTLFilePath is not None and forceNoPauses == False:
    # Show point in 3D space
        print('==== Showing STL and wand tip in 3D navigator ...')
        vtkSegment = fio.readSTL(segSTLFilePath)
        actorSegment = vtkh.createVTKActor(vtkSegment)
        vtkWandTip = vtkh.createSphereVTKData(tipMimicsAvg, 3)
        actorWandTip = vtkh.createVTKActor(vtkWandTip, color=(1,0,0))
        actors = [actorSegment, actorWandTip]
        vtkh.showVTKActors(actors)
        
    print('==== Finished')
    
    return tipMimicsAvg

