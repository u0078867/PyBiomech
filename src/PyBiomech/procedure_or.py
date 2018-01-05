"""
.. module:: procedure_or
   :synopsis: helper module for procedures used with Oxford-Rig (IORT UZLeuven)

"""

import numpy as np

import fio, kine, kine_or, vtkh, ligaments as liga, contacts

import re
from itertools import groupby


def expressOptoWandTipToMimicsRefFrame(
                                        filePathC3D, 
                                        filePathMimics, 
                                        wandTipName, 
                                        refSegment,
                                        wandParams = None,
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
        'setMarkersZeroValuesToNaN': True,
        'removeSegmentNameFromMarkerNames': True,
    })['markers']
    
    # Calculate wand tip in Optoelectronic reference frame
    print('==== Reconstructing wand tip ...')
    if wandParams is None:
        tipOpto = kine_or.getPointerTipOR(markersOpto, verbose=verbose)
    else:
        tipOpto = kine.nonCollinear5PointsStylusFun(markersOpto, wandParams, verbose=verbose)
    
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
    else:
        mkrList = refSegment[:]
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
            
            


def calculateKneeSegmentsPoses(
                            markersOpto,
                            markersLoc,
                            verbose = False,
                            vtkFemur = None,
                            vtkTibia = None,
                            vtkPatella = None,
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
    
    femurAvailable = False
    mkrList = ["mF1", "mF2", "mF3", "mF4"]
    if set(mkrList) <= set(markersLoc.keys()):
        femurAvailable = True
        print('==== ---- Femur markers found')
        R1, T1, info1 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
        RT1 = kine.composeRotoTranslMatrix(R1, T1)
    
    tibiaAvailable = False
    mkrList = ["mT1", "mT2", "mT3", "mT4"]
    if set(mkrList) <= set(markersLoc.keys()):
        tibiaAvailable = True
        print('==== ---- Tibia markers found')
        R2, T2, info2 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
        RT2 = kine.composeRotoTranslMatrix(R2, T2)
    
    patellaAvailable = False
    mkrList = ["mP1", "mP2", "mP3", "mP4"]
    if set(mkrList) <= set(markersLoc.keys()):
        patellaAvailable = True
        print('==== ---- Patella markers found')
        R3, T3, info3 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
        RT3 = kine.composeRotoTranslMatrix(R3, T3)
    
    results = {}
    if femurAvailable:
        results['femur_pose'] = RT1
        results['femur_pose_reconstruction_info'] = info1
    if tibiaAvailable:
        results['tibia_pose'] = RT2
        results['tibia_pose_reconstruction_info'] = info2
    if patellaAvailable:
        results['patella_pose'] = RT3
        results['patella_pose_reconstruction_info'] = info3
    
    Nf = RT1.shape[0]

    iRange = range(Nf)
    if sceneFrames is not None:
        iRange = sceneFrames
        
    if saveScene and vtkFemur is not None and not femurAvailable:
        
        raise Exception('"saveScene = True" option requires femur data to be available if input "vtkFemur" is provided')
    
    if saveScene and vtkTibia is not None and not tibiaAvailable:
        
        raise Exception('"saveScene = True" option requires tibia data to be available if input "vtkTibia" is provided')
        
    if saveScene and vtkPatella is not None and not patellaAvailable:
        
        raise Exception('"saveScene = True" option requires patella data to be available if input "vtkPatella" is provided')
    
    for i in iRange:
        
        if saveScene:
        
            print('==== ---- Saving scene for time frame %d' % (i))
        
            if vtkFemur is not None:
            
                vtkFemur2 = vtkh.reposeVTKData(vtkFemur, RT1[i,...])
                
            if vtkTibia is not None:
                
                vtkTibia2 = vtkh.reposeVTKData(vtkTibia, RT2[i,...])
                
            if vtkPatella is not None:
            
                vtkPatella2 = vtkh.reposeVTKData(vtkPatella, RT3[i,...])
        
                
            actors = []
            names = []
            
            if vtkFemur is not None:
            
                actors.append(vtkh.createVTKActor(vtkFemur2))
                names.append('femur')
                
            if vtkTibia is not None:
            
                actors.append(vtkh.createVTKActor(vtkTibia2))
                names.append('tibia')
                
            if vtkPatella is not None:
            
                actors.append(vtkh.createVTKActor(vtkPatella2))
                names.append('patella')
            
            scene = vtkh.createScene(actors)
            
            for fmt in sceneFormats:
                vtkh.exportScene(scene, outputDirSceneFile + ('/poses_tf_%05d' % i), ext=fmt, names=names)
                
    
    return results
    

def calculateKneeKinematics(
                            RT1,
                            RT2,
                            landmarksLoc,
                            side
                            ):
                                
    allALs = {}

    # express FEMUR anatomical landmarks in lab reference frame
    print('==== Expressing femur anatomical landmarks in Optoelectronic reference frame ...')
    ALs = {m: np.array(landmarksLoc[m]) for m in ["FHC","FKC","FMCC","FLCC"]}
    allALs.update(kine.changeMarkersReferenceFrame(ALs, RT1))

    # calculate FEMUR anatomical reference frame
    Ra1, Oa1 = kine_or.femurOR(allALs, s=side)

    # express TIBIA anatomical landmarks in lab reference frame
    print('==== Expressing tibia anatomical landmarks in Optoelectronic reference frame ...')
    ALs = {m: np.array(landmarksLoc[m]) for m in ["TAC","TKC","TMCC","TLCC"]}
    allALs.update(kine.changeMarkersReferenceFrame(ALs, RT2))

    # calculate TIBIA anatomical reference frame
    Ra2, Oa2 = kine_or.tibiaOR(allALs, s=side)

    # calculate KNEE kinematics
    print('==== Calculating knee kinematics ...')
    angles = kine.getJointAngles(Ra1, Ra2, R2anglesFun=kine_or.gesOR, funInput='segmentsR', s=side)
    transl = kine.getJointTransl(Ra1, Ra2, Oa1, Oa2, T2translFun=kine_or.gesTranslOR)
    
    results = {}
    results["extension"] = angles[:,0]
    results["adduction"] = angles[:,1]
    results["intrarotation"] = angles[:,2]
    results["ML"] = transl[:,0]
    results["AP"] = transl[:,1]
    results["IS"] = transl[:,2]
    results["landmarks"] = allALs
    
    return results
    
    
    


def calculateKneeLigamentsData(
                                RT1,
                                RT2,
                                insertionsLoc,
                                frames = None,
                                ligaNames = ['MCL', 'LCL'],
                                tibiaPlateauMedEdgeSplineLoc = None,
                                ligaModels = ['straight','Blankevoort_1991'],
                                vtkFemur = None,
                                vtkTibia = None,
                                Marai2004Params = {},
                                saveScene = False,
                                sceneFormats = ['vtm'],
                                outputDirSceneFile = None,
                              ):
                              
    print('==== Creating spline for tibial medial edge plateau ...')
    if tibiaPlateauMedEdgeSplineLoc is not None:
        tibiaPlateauMedEdgeParamsSplineLoc = vtkh.createParamSpline(tibiaPlateauMedEdgeSplineLoc)
    else:
        tibiaPlateauMedEdgeParamsSplineLoc = None
    
    # initialize all insertion points dict
    allPointsIns = {}

    # Create dependency for ligaments
    usedIns1 = []
    usedIns2 = []
    availableLigaNames = []
    depsIns = {}
    if "MCL" in ligaNames:
        if "FMCL" in insertionsLoc and "TMCL" in insertionsLoc:
            usedIns1.append("FMCL")
            usedIns2.append("TMCL")
            availableLigaNames.append("MCL")
            depsIns['MCL'] = ['FMCL', 'TMCL']
    if "LCL" in ligaNames:
        if "FLCL" in insertionsLoc and "TLCL" in insertionsLoc:
            usedIns1.append("FLCL")
            usedIns2.append("TLCL")
            availableLigaNames.append("LCL")
            depsIns['LCL'] = ['FLCL', 'TLCL']

    # express FEMUR insertion points in lab reference frame
    print('==== Expressing femur ligament insertions in Optoelectronic reference frame ...')
    pointsInsLoc = {m: np.array(insertionsLoc[m]) for m in usedIns1}
    allPointsIns.update(kine.changeMarkersReferenceFrame(pointsInsLoc, RT1))

    # express TIBIA insertion points in lab reference frame
    print('==== Expressing tibia ligament insertions in Optoelectronic reference frame ...')
    pointsInsLoc = {m: np.array(insertionsLoc[m]) for m in usedIns2}
    allPointsIns.update(kine.changeMarkersReferenceFrame(pointsInsLoc, RT2))
    
    # calculate liga paths and lengths
    print('==== Calculating knee ligaments data ...')
    ligaPaths = {}
    ligaLengths = {}
    
    Nf = RT1.shape[0]

    iRange = range(Nf)
    if frames is not None:
        iRange = frames
    
    for i in iRange:
        
        print('==== ---- Processing time frame %d' % (i))
        
        if saveScene:
        
            if vtkFemur is not None and vtkTibia is not None:
            
                vtkFemur2 = vtkh.reposeVTKData(vtkFemur, RT1[i,...])
                vtkTibia2 = vtkh.reposeVTKData(vtkTibia, RT2[i,...])
        
            else:
                
                raise Exception('"saveScene = True" option requires input "vtkFemur", "vtkTibia"')
    
        for ligaName in availableLigaNames:
            
            if ligaName not in ligaPaths:
                ligaPaths[ligaName] = {}
                
            if ligaName not in ligaLengths:
                ligaLengths[ligaName] = {}
            
            insNames = depsIns[ligaName]
            
            pIns2 = np.array((allPointsIns[insNames[0]][i,:], allPointsIns[insNames[1]][i,:]))
            
            if 'straight' in ligaModels:
                
                if 'straight' not in ligaPaths[ligaName]:
                    ligaPaths[ligaName]['straight'] = Nf * [[]]
                    
                if 'straight' not in ligaLengths[ligaName]:
                    ligaLengths[ligaName]['straight'] = Nf * [np.nan]
    
                # calculate path
                ligaPathA = pIns2.copy()
                
                ligaLengthA = liga.calcLigaLength(ligaPathA)
                
                ligaPaths[ligaName]['straight'][i] = ligaPathA
                
                ligaLengths[ligaName]['straight'][i] = ligaLengthA
            
            if 'Blankevoort_1991' in ligaModels:
                
                if 'Blankevoort_1991' not in ligaPaths[ligaName]:
                    ligaPaths[ligaName]['Blankevoort_1991'] = Nf * [[]]
                    
                if 'Blankevoort_1991' not in ligaLengths[ligaName]:
                    ligaLengths[ligaName]['Blankevoort_1991'] = Nf * [np.nan]
    
                if tibiaPlateauMedEdgeParamsSplineLoc is not None:
        
                    # repose spline
                    tibiaPlateauMedEdgeParamsSpline = vtkh.reposeSpline(tibiaPlateauMedEdgeParamsSplineLoc, RT2[i,...])
        
                    # 2-lines model with shortest possible path if touching tibial edge
                    dummy, ligaPathB = liga.ligamentPathBlankevoort1991(pIns2, tibiaPlateauMedEdgeParamsSpline)
                    
                    # Calculate length
                    ligaLengthB = liga.calcLigaLength(ligaPathB)
                    
                    ligaPaths[ligaName]['Blankevoort_1991'][i] = ligaPathB
                
                    ligaLengths[ligaName]['Blankevoort_1991'][i] = ligaLengthB
                    
                else:
                    
                    raise Exception('"Blankevoort_1991" methods requires input "tibiaPlateauMedEdgeSplineLoc"')
                    
            if 'Marai_2004' in ligaModels:
                
                if 'Marai_2004' not in ligaPaths[ligaName]:
                    ligaPaths[ligaName]['Marai_2004'] = Nf * [[]]
                    
                if 'Marai_2004' not in ligaLengths[ligaName]:
                    ligaLengths[ligaName]['Marai_2004'] = Nf * [np.nan]
                
                if vtkFemur is not None and vtkTibia is not None and Marai2004Params is not None:
                    
                    vtkFemur2 = vtkh.reposeVTKData(vtkFemur, RT1[i,...])
                    
                    vtkTibia2 = vtkh.reposeVTKData(vtkTibia, RT2[i,...])
                    
                    pars = Marai2004Params
                    
                    dummy, ligaPathC = liga.ligamentPathMarai2004(pIns2, vtkFemur2, vtkTibia2, **pars)
                    
                    # Calculate length
                    ligaLengthC = liga.calcLigaLength(ligaPathC)
                    
                    ligaPaths[ligaName]['Marai_2004'][i] = ligaPathC
                
                    ligaLengths[ligaName]['Marai_2004'][i] = ligaLengthC
                    
                else:
                    
                    raise Exception('"Marai_2004" methods requires input "vtkFemur", "vtkTibia"')
                    
        if saveScene:
            
            actors = []
            names = []
            
            actors.append(vtkh.createVTKActor(vtkFemur2))
            names.append('femur')
            
            actors.append(vtkh.createVTKActor(vtkTibia2))
            names.append('tibia')
            
            if tibiaPlateauMedEdgeParamsSplineLoc is not None:
            
                tibiaPlateauMedEdgePoints = vtkh.evalSpline(tibiaPlateauMedEdgeParamsSpline, np.arange(0, 1.01, 0.01))
                
                vtkTibiaPlateauMedEdgeLine = vtkh.createLineVTKData(tibiaPlateauMedEdgePoints, [255, 0, 0])
                
                actors.append(vtkh.createVTKActor(vtkTibiaPlateauMedEdgeLine))
                names.append('tibia_medial_plateau')
            
            for ligaName in ligaPaths:
                
                for model in ligaPaths[ligaName]:
                    
                    vtkLigaLine = vtkh.createLineVTKData(ligaPaths[ligaName][model][i], [255, 0, 0])
            
                    actors.append(vtkh.createVTKActor(vtkLigaLine))
                    names.append(ligaName + '_model_' + model)
            
            scene = vtkh.createScene(actors)
            
            for fmt in sceneFormats:
                vtkh.exportScene(scene, outputDirSceneFile + ('/ligaments_tf_%05d' % i), ext=fmt, names=names)

    
    # store results
    results = {}
    results["paths"] = ligaPaths
    results["lengths"] = ligaLengths
    return results





def calculateKneeContactsData(
                                RT1,
                                RT2,
                                vtkFemur,
                                vtkTibia,
                                frames = None,
                                femurDecimation = None,
                                tibiaDecimation = None,
                                saveScene = False,
                                sceneFormats = ['vtm'],
                                outputDirSceneFile = None,
                              ):

    if femurDecimation is not None:
        
        vtkFemur_ = vtkh.decimateVTKData(vtkFemur, femurDecimation)
        
    else:
        
        vtkFemur_ = vtkFemur
        
        
    if tibiaDecimation is not None:
        
        vtkTibia_ = vtkh.decimateVTKData(vtkTibia, tibiaDecimation)
        
    else:
        
        vtkTibia_ = vtkTibia

    print('==== Calculating knee contact data ...')

    Nf = RT1.shape[0]
    
    contactData = {}
    contactData['femur'] = {}
    contactData['femur']['points'] = Nf * [[]]
    contactData['femur']['distances'] = Nf * [[]]
    contactData['tibia'] = {}
    contactData['tibia']['points'] = Nf * [[]]
    contactData['tibia']['distances'] = Nf * [[]]
    
    iRange = range(Nf)
    if frames is not None:
        iRange = frames
    
    for i in iRange:
    
        print('==== ---- Processing time frame %d' % (i))

        vtkFemur2 = vtkh.reposeVTKData(vtkFemur_, RT1[i,...])
                    
        vtkTibia2 = vtkh.reposeVTKData(vtkTibia_, RT2[i,...])
        
        vtkFemur2Sliced, vtkTibia2Sliced = contacts.calculateBonesContactAnalysisROIonScaledBB(vtkFemur2, vtkTibia2, 3*[1.2], 3*[1.2])
        
        vtkFemurDistance, vtkTibiaDistance = contacts.calculateBonesContactData(vtkFemur2Sliced, vtkTibia2Sliced)
        
        if saveScene:
            
            names = []
            
            vtkFemurActor = vtkh.createVTKActor(vtkFemur2)
            names.append('femur')
            
            vtkTibiaActor = vtkh.createVTKActor(vtkTibia2)
            names.append('tibia')
            
            vtkFemurDistanceActor = vtkh.createVTKActor(vtkFemurDistance, scalarRange=vtkFemurDistance.GetScalarRange())
            names.append('femur_distance_field')
        
            vtkTibiaDistanceActor = vtkh.createVTKActor(vtkTibiaDistance, scalarRange=vtkTibiaDistance.GetScalarRange())
            names.append('tibia_distance_field')
            
            vtkFemurDistanceContour = vtkh.createContourVTKData(vtkFemurDistance, 20)
            
            vtkTibiaDistanceContour = vtkh.createContourVTKData(vtkTibiaDistance, 20)
            
            vtkFemurDistanceContourActor = vtkh.createVTKActor(vtkFemurDistanceContour, 
                                                               presets='contour', 
                                                               color=(0.2, 0.2, 0.2),
                                                               lineWidth=2, 
                                                               scalarRange=vtkFemurDistance.GetScalarRange()
                                                              )
            names.append('femur_distance_field_isolines')
                                                              
            vtkTibiaDistanceContourActor = vtkh.createVTKActor(vtkTibiaDistanceContour, 
                                                               presets='contour', 
                                                               color=(0.2, 0.2, 0.2),
                                                               lineWidth=2, 
                                                               scalarRange=vtkTibiaDistance.GetScalarRange()
                                                              )
            names.append('tibia_distance_field_isolines')
            
            scene = vtkh.createScene([
                vtkFemurActor,
                vtkTibiaActor,
                vtkFemurDistanceActor,
                vtkTibiaDistanceActor,
                vtkFemurDistanceContourActor,
                vtkTibiaDistanceContourActor
            ])
            #vtkh.showScene(scene)
            
            for fmt in sceneFormats:
                vtkh.exportScene(scene, outputDirSceneFile + ('/contact_tf_%05d' % i), ext=fmt, names=names)
        
        points, distances = vtkh.VTKScalarData2Numpy(vtkFemurDistance)
        contactData['femur']['points'][i] = points
        contactData['femur']['distances'][i] = distances
        
        points, distances = vtkh.VTKScalarData2Numpy(vtkTibiaDistance)
        contactData['tibia']['points'][i] = points
        contactData['tibia']['distances'][i] = distances
        
        
    # store results
    results = contactData
    return results
    
    
    
    
def assembleKneeDataGranular(
                                markers=None, 
                                poses=None, 
                                kine=None, 
                                liga=None, 
                                contact=None
                            ):
                                
    print('==== Assemblying knee data ...')
    
    Nf = poses['femur_pose'].shape[0]
    
    data = Nf * [None]
    
    for i in xrange(Nf):
        
        row = []
        
        # Assemble markers
        
        if markers is not None:
        
            for m in markers:
                
                item = {
                    "description": ("%s global position (mm)" % m),
                    "value": markers[m][i,:],
                    "ID": m,
                    "type": "point",
                    "by": "C3D reader"
                }
                row.append(item)
                
        if kine is not None:
            
            for m in kine["landmarks"]:
                
                item = {
                    "description": ("%s global position (mm)" % m),
                    "value": kine["landmarks"][m][i,:],
                    "ID": m,
                    "type": "point",
                    "by": "KneeKine"
                }
                row.append(item)
            
        # Assemble poses
            
        if poses is not None:
            
            item = {
                "description": "Pose from femur technical to lab reference frame",
                "value": poses['femur_pose'][i,...].squeeze(),
                "ID": 'femurPose',
                "type": "pose",
                "by": "KneeKine"
            }
            row.append(item)
            
            item = {
                "description": "Pose from tibia technical to lab reference frame",
                "value": poses['tibia_pose'][i,...].squeeze(),
                "ID": 'tibiaPose',
                "type": "pose",
                "by": "KneeKine"
            }
            row.append(item)
        
        # Assemble ligament data
        
        if liga is not None:
        
            for ligaName in liga['paths']:
                
                for ligaModel in liga['paths'][ligaName]:
                    
                    item = {
                        "description+": ("%s global path position (mm)" % ligaName),
                        "value": liga['paths'][ligaName][ligaModel][i],
                        "ID": ('%sPath_%s' % (ligaName, ligaModel)),
                        "type": "path",
                        "by": "LigaPath"
                    }
                    row.append(item)                
                
                
        
        data[i] = row
            
            
    return data
        
        
        
def assembleKneeDataAsIsNoMetadata(
                                    markers=None, 
                                    poses=None, 
                                    kine=None, 
                                    liga=None, 
                                    contact=None
                                  ):
                                      
    print('==== Assemblying knee data ...')
    
    data = {}
    
    if markers is not None:
    
        data['markers'] = markers
        
        if kine is not None:
            
            data['markers'].update(kine['landmarks'])
            
    if poses is not None:
    
        data['poses'] = poses
        
    if kine is not None:
    
        data['kine'] = kine

    if liga is not None:
    
        data['liga'] = liga
        
    if contact is not None:
    
        data['contact'] = contact
    
    return data
    
    