"""
.. module:: procedure_or
   :synopsis: helper module for procedures used with Oxford-Rig (IORT UZLeuven)

"""

import numpy as np
import fio, kine, kine_or, vtkh
import re
from itertools import groupby


def expressOptoWandTipToMimicsRefFrame(
                                        filePathC3D, 
                                        filePathMimics, 
                                        wandTipName, 
                                        refSegment,
                                        filePathNewC3D = None,
                                        reduceAs = 'avg_point',
                                        segSTLFilePath = None,
                                        verbose = False,
                                        showNavigator = True,
                                        forceNoPauses = False
                                        ):
    """This procedure reconstruct the tip of the Ofxord-rig wand in the Mimics
    reference frame in which the technical markers mXY (X = T|F; Y = 1-4)
    are expressed.

    Parameters
    ----------
    filePathC3D : str
        Path of the C3D file where the wand is pointing a point. Necessary
        markers are:
        
        - 'WN', 'WW', 'WE', 'WM', 'WS': wand markers;
        - 'mX1' - 'mX4': segment fixed markers;
        if 'refSegment' is ``'femur'``, then X = F;
        if 'refSegment' is ``'tibia'``, then X = T;
    
    filePathMimics : str
        Path of the Mimics TXT file containing point coordinates of in the 
        Mimics reference frame. The following points are necessary:
        
        - 'mX1' - 'mX4': segment fixed markers;
        if 'refSegment' is ``'femur'``, then X = F;
        if 'refSegment' is ``'tibia'``, then X = T;        
    
    wandTipName : str
        Name of the wand tip. It is used when creating the new C3D.
        
    refSegment : str
        Name of the segment fixed to the poiint indicated by the wand.
        It can be ``'femur'`` or ``'tibia'``.
        
    filePathNewC3D : str
        If not None, the procedure will create a new C3D file containing the
        wand tip, and all the points from the source C3D file, at the path
        indicated with this parameter.
    
    reduceAs : str
        If ``'avg_point'``, the coordinates of the wand tip at each frame
        instant are averaged.
        If None, ``tipMimicsReduced`` output will be None.
        
    segSTLFilePath : str
        Path of the STL file of the bone segment fixed to the technical markers
        mXY. If provided, it will be shown in the 3D navigator.
        
    verbose : bool
        If True, more info is shown in the console.
        
    showNavigator : bool
        If True, it will show a 3D navigator (and stop the execution of the 
        code), showing:
        
        - the STL file, if path is provided;
        - the reconstructed point, if ``reduceAs='avg_point'``;
        - the reconstructed point as line, if ``reduceAs=None``;
    
    forceNoPauses : bool
        If True, the function will not stop for operations such as 3D navigation.
    

    Returns
    -------
    tipMimics : np.array
        N x 3 array, where N is the time frames number in the C3D file.
        It represents the wand tip for each time frame.

    tipMimicsReduced : np.ndarray
        3-elem array, representing the coordinates of the wand tip
        (see ``reduceAs`` parameter).

    """


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
        data['markers']['data'][wandTipName] = tipOpto
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
    print('==== Inverting pose ...')
    mRTo = kine.inv2(oRTm)
    
    # Express tip in Mimics reference frame
    print('==== Expressing tip in Mimics reference frame ...')
    tipMimics = kine.changeMarkersReferenceFrame({'tip': tipOpto}, mRTo)['tip']
    tipMimicsAvg = np.mean(tipMimics, axis=0)
    
    # Reduce points
    if reduceAs == 'avg_point':
        tipMimicsReduced = tipMimicsAvg
    else:
        tipMimicsReduced = None
    
    if showNavigator == True and forceNoPauses == False:
    # Show point in 3D space
        print('==== Showing VTK 3D navigator ...')
        actors = []
        if segSTLFilePath is not None:
            vtkSegment = fio.readSTL(segSTLFilePath)
            actorSegment = vtkh.createVTKActor(vtkSegment)
            actors.append(actorSegment)
        # Create wand tip actor
        if reduceAs == 'avg_point':
            vtkWandTip = vtkh.createSphereVTKData(tipMimicsAvg, 3)
            actorWandTip = vtkh.createVTKActor(vtkWandTip, color=(1,0,0))
        else:
            vtkWandTip = vtkh.createLineVTKData(tipMimics, (255,0,0))
            actorWandTip = vtkh.createVTKActor(vtkWandTip)
        actors.append(actorWandTip)
        # Show actors in navigator
        vtkh.showVTKActors(actors)
        
    print('==== Finished')
    
    return tipMimics, tipMimicsReduced



def groupListBy(L, T, kf):
    """Given a list of strings, for each one find tokens by using a regex
    pattern and group string by a key created by using tokens.
    Normally used group file paths.

    Parameters
    ----------
    L : list
        List of strings to be grouped.
    
    T : str
        Pattern to match (see ``'re.search()'``)     
    
    kf : fun
        Function whose input is a list of tokens, and the output is the key
        to use for grouping.

    Returns
    -------
    d : dict
        Dictionary where each key is the one generated by the kf function, and
        valus is the list of relative tokens.

    """
    d = {}
    things = [m.groups() for m in (re.search(T, l) for l in L) if m]
    for key, group in groupby(things, kf):
        d[key] = []
        for thing in group:
            d[key].append(thing)
    return d
            
            
            