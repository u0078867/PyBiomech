"""
.. module:: procedure_or
   :synopsis: helper module for procedures used with Oxford-Rig (IORT UZLeuven)

"""

import numpy as np

import fio, mplh, kine, kine_or, vtkh, ligaments as liga, contacts, spine

import re
from itertools import groupby

import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
    oRm, oTm, info = kine.rigidBodySVDFun(markersOpto, mkrList, args)
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
    femurError = False
    mkrList = ["mF1", "mF2", "mF3", "mF4"]
    if set(mkrList) <= set(markersLoc.keys()):
        femurAvailable = True
        print('==== ---- Femur markers found')
        try:
            R1, T1, info1 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
            RT1 = kine.composeRotoTranslMatrix(R1, T1)
        except Exception as e:
            print('Impossible to calculate femur pose')
            print(e)
            femurError = True
    
    tibiaAvailable = False
    tibiaError = False
    mkrList = ["mT1", "mT2", "mT3", "mT4"]
    if set(mkrList) <= set(markersLoc.keys()):
        tibiaAvailable = True
        print('==== ---- Tibia markers found')
        try:
            R2, T2, info2 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
            RT2 = kine.composeRotoTranslMatrix(R2, T2)
        except Exception as e:
            print('Impossible to calculate tibia pose')
            print(e)
            tibiaError = True
    
    patellaAvailable = False
    patellaError = False
    mkrList = ["mP1", "mP2", "mP3", "mP4"]
    if set(mkrList) <= set(markersLoc.keys()):
        patellaAvailable = True
        print('==== ---- Patella markers found')
        try:
            R3, T3, info3 = kine.rigidBodySVDFun(markersOpto, mkrList, args)
            RT3 = kine.composeRotoTranslMatrix(R3, T3)
        except Exception as e:
            print('Impossible to calculate tibia pose')
            print(e)
            patellaError = True
    
    results = {}
    if femurAvailable and not femurError:
        results['femur_pose'] = RT1
        results['femur_pose_reconstruction_info'] = info1
    if tibiaAvailable and not tibiaError:
        results['tibia_pose'] = RT2
        results['tibia_pose_reconstruction_info'] = info2
    if patellaAvailable and not patellaError:
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
                                tibiaPlateauLatEdgeSplineLoc = None,
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
        
    print('==== Creating spline for tibial lateral edge plateau ...')
    if tibiaPlateauLatEdgeSplineLoc is not None:
        tibiaPlateauLatEdgeParamsSplineLoc = vtkh.createParamSpline(tibiaPlateauLatEdgeSplineLoc)
    else:
        tibiaPlateauLatEdgeParamsSplineLoc = None
    
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
    if "MCLa" in ligaNames:
        if "FMCLa" in insertionsLoc and "TMCLa" in insertionsLoc:
            usedIns1.append("FMCLa")
            usedIns2.append("TMCLa")
            availableLigaNames.append("MCLa")
            depsIns['MCLa'] = ['FMCLa', 'TMCLa']
    if "MCLm" in ligaNames:
        if "FMCLm" in insertionsLoc and "TMCLm" in insertionsLoc:
            usedIns1.append("FMCLm")
            usedIns2.append("TMCLm")
            availableLigaNames.append("MCLm")
            depsIns['MCLm'] = ['FMCLm', 'TMCLm']
    if "MCLp" in ligaNames:
        if "FMCLp" in insertionsLoc and "TMCLp" in insertionsLoc:
            usedIns1.append("FMCLp")
            usedIns2.append("TMCLp")
            availableLigaNames.append("MCLp")
            depsIns['MCLp'] = ['FMCLp', 'TMCLp']
    if "LCL" in ligaNames:
        if "FLCL" in insertionsLoc and "TLCL" in insertionsLoc:
            usedIns1.append("FLCL")
            usedIns2.append("TLCL")
            availableLigaNames.append("LCL")
            depsIns['LCL'] = ['FLCL', 'TLCL']
    if "LCLa" in ligaNames:
        if "FLCLa" in insertionsLoc and "TLCLa" in insertionsLoc:
            usedIns1.append("FLCLa")
            usedIns2.append("TLCLa")
            availableLigaNames.append("LCLa")
            depsIns['LCLa'] = ['FLCLa', 'TLCLa']
    if "LCLm" in ligaNames:
        if "FLCLm" in insertionsLoc and "TLCLm" in insertionsLoc:
            usedIns1.append("FLCLm")
            usedIns2.append("TLCLm")
            availableLigaNames.append("LCLm")
            depsIns['LCLm'] = ['FLCLm', 'TLCLm']
    if "LCLp" in ligaNames:
        if "FLCLp" in insertionsLoc and "TLCLp" in insertionsLoc:
            usedIns1.append("FLCLp")
            usedIns2.append("TLCLp")
            availableLigaNames.append("LCLp")
            depsIns['LCLp'] = ['FLCLp', 'TLCLp']

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
                    
                if 'MCL' in ligaName:
                    
                    if tibiaPlateauMedEdgeParamsSplineLoc is None:
                    
                        raise Exception('"Blankevoort_1991" methods requires input "tibiaPlateauMedEdgeSplineLoc" for MCL(a,m,p)')
                        
                    # repose spline
                    tibiaPlateauMedEdgeParamsSpline = vtkh.reposeSpline(tibiaPlateauMedEdgeParamsSplineLoc, RT2[i,...])
    
                    # 2-lines model with shortest possible path if touching tibial edge
                    dummy, ligaPathB = liga.ligamentPathBlankevoort1991(pIns2, tibiaPlateauMedEdgeParamsSpline)
                    
                elif 'LCL' in ligaName:
                    
                    if tibiaPlateauLatEdgeParamsSplineLoc is None:
                    
                        raise Exception('"Blankevoort_1991" methods requires input "tibiaPlateauLatEdgeSplineLoc" for LCL(a,m,p)')
                        
                    # repose spline
                    tibiaPlateauLatEdgeParamsSpline = vtkh.reposeSpline(tibiaPlateauLatEdgeParamsSplineLoc, RT2[i,...])
    
                    # 2-lines model with shortest possible path if touching tibial edge
                    dummy, ligaPathB = liga.ligamentPathBlankevoort1991(pIns2, tibiaPlateauLatEdgeParamsSpline)
                
                # Calculate length
                ligaLengthB = liga.calcLigaLength(ligaPathB)
                
                ligaPaths[ligaName]['Blankevoort_1991'][i] = ligaPathB
            
                ligaLengths[ligaName]['Blankevoort_1991'][i] = ligaLengthB
                    
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

            if tibiaPlateauLatEdgeParamsSplineLoc is not None:
            
                tibiaPlateauLatEdgePoints = vtkh.evalSpline(tibiaPlateauLatEdgeParamsSpline, np.arange(0, 1.01, 0.01))
                
                vtkTibiaPlateauLatEdgeLine = vtkh.createLineVTKData(tibiaPlateauLatEdgePoints, [255, 0, 0])
                
                actors.append(vtkh.createVTKActor(vtkTibiaPlateauLatEdgeLine))
                names.append('tibia_lateral_plateau')
            
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
    
    
    
    

def performSpineAnalysis(
                        markers1,
                        markers2,
                        singleMarkersDescFile,
                        segmentsDescFile,
                        clustersDescDir,
                        spinePointNames,
                        resultsDir,
                        savePlots=True
                        ):
                            
    # calculate THORAX anatomical reference frame for acquisition system 1
    markers1Tho = {}
    markers1Tho['IJ'] = markers1['CLAV']
    markers1Tho['PX'] = markers1['STRN']
    markers1Tho['C7'] = markers1['C7']
    markers1Tho['T8'] = markers1['T9']
    RTho1, OTho1 = kine.thoraxPoseISB(markers1Tho)
    RTTho1 = kine.composeRotoTranslMatrix(RTho1, OTho1)
    RTTho1i = kine.inv2(RTTho1)
    
    # calculate PELVIS anatomical reference frame for acquisition system 1
    RPel1, OPel1 = kine.pelvisPoseISB(markers1, s='R')
    RTPel1 = kine.composeRotoTranslMatrix(RPel1, OPel1)
    RTPel1i = kine.inv2(RTPel1)
    
    # calculate THORAX anatomical reference frame for acquisition system 2
    markers2Tho = {}
    markers2Tho['IJ'] = markers2['CLAV']
    markers2Tho['PX'] = markers2['STRN']
    markers2Tho['C7'] = markers2['C7']
    markers2Tho['T8'] = markers2['T9']
    RTho2, OTho2 = kine.thoraxPoseISB(markers2Tho)
    RTTho2 = kine.composeRotoTranslMatrix(RTho2, OTho2)
    
    # calculate PELVIS anatomical reference frame for acquisition system 2
    RPel2, OPel2 = kine.pelvisPoseISB(markers2, s='R')
    RTPel2 = kine.composeRotoTranslMatrix(RPel2, OPel2)
    RTPel2i = kine.inv2(RTPel2)
    
    # Handles single markers
    markers2New = {}
    singleMarkersInfo = fio.readStringListMapFile(singleMarkersDescFile)
    for m in singleMarkersInfo:
        markerName = m
        segment = singleMarkersInfo[m][0]
        
        # Express real marker in anatomical reference frame 
        if segment == 'thorax':
            RT1i = RTTho1i
        elif segment == 'pelvis':
            RT1i = RTPel1i
        else:
            raise Exception('segment must be one of: thorax, pelvis')
        targetMarkers = {}
        targetMarkers[markerName + ' base'] = markers1[markerName + ' base']
        targetMarkers['True ' + markerName] = markers1['True ' + markerName]
        targetMarkersLoc = kine.changeMarkersReferenceFrame(targetMarkers, RT1i)
        
        #
        if segment == 'thorax':
            RT2 = RTTho2
        elif segment == 'pelvis':
            RT2 = RTPel2
        else:
            raise Exception('segment must be one of: thorax, pelvis')
        markers2New.update(kine.changeMarkersReferenceFrame(targetMarkersLoc, RT2))
        
    
    # Handles clusters
    segmentsInfo = fio.readStringListMapFile(segmentsDescFile)
    print segmentsInfo
    
    for s in segmentsInfo:
        segmentName = s
        clusterType = segmentsInfo[s][0]
        print('%s %s' % (segmentName, clusterType))
        
        # Read data for cluster connected to current segment
        clusterInfo = fio.readStringListMapFile(os.path.join(clustersDescDir, clusterType + '.txt'))
        clusterInfo = {m: np.array(clusterInfo[m]) for m in clusterInfo}
        
        # Modify cluster marker names with real ones
        clusterInfoSpec = clusterInfo.copy()
        subs = segmentsInfo[s][1:]
        clusterBaseName = clusterBaseNameSpec = 'Base'
        for sub in subs:
            s = sub.split(':')
            if s[0] == clusterBaseName:
                clusterBaseNameSpec = s[1]
            del clusterInfoSpec[s[0]]
            clusterInfoSpec[s[1]] = clusterInfo[s[0]]
        print(clusterInfoSpec)
        
        # SVD for acquisition system 1
        args = {}
        markersLoc = clusterInfoSpec.copy()
        args['mkrsLoc'] = markersLoc
        args['verbose'] = True
        mkrList = markersLoc.keys()
        R1, T1, infoSVD1 = kine.rigidBodySVDFun(markers1, mkrList, args)
        RT1 = kine.composeRotoTranslMatrix(R1, T1)
        
        # Express cluster base and real marker in cluster reference frame
        RT1i = kine.inv2(RT1)
        targetMarkers = {}
        targetMarkers[clusterBaseNameSpec + ' 1'] = markers1[clusterBaseNameSpec]
        targetMarkers['True ' + segmentName] = markers1['True ' + segmentName]
        targetMarkersLoc = kine.changeMarkersReferenceFrame(targetMarkers, RT1i)
        
        # SVD for acquisition system 2
        clusterBaseMarkerLoc = clusterInfoSpec[clusterBaseNameSpec]
        args = {}
        markersLoc = clusterInfoSpec.copy()
        del markersLoc[clusterBaseNameSpec]
        args['mkrsLoc'] = markersLoc
        args['verbose'] = True
        mkrList = markersLoc.keys()
        R2, T2, infoSVD2 = kine.rigidBodySVDFun2(markers2, mkrList, args)
        RT2 = kine.composeRotoTranslMatrix(R2, T2)
        
        # Copy and rename cluster base marker to be expressed in system 2 in global reference frame
        targetMarkersLoc[clusterBaseNameSpec + ' 2'] = clusterBaseMarkerLoc
        
        # Update list of markers for system 2 in global reference frame
        markers2New.update(kine.changeMarkersReferenceFrame(targetMarkersLoc, RT2))
    
    # Express spine points in pelvis reference frame
    spinePoints = {m: markers2New[m] for m in spinePointNames}
    spinePointsPel = kine.changeMarkersReferenceFrame(spinePoints, RTPel2i)
    
    # Merge spine points in one array
    spineData = np.stack([spinePointsPel[m] for m in spinePointNames], axis=2)  # Nf x 3 x Np
    
    # Init results
    res = {}
    res['newMarkers'] = markers2New
    sup, inf = spinePointNames[:-1], spinePointNames[1:]
    spineAngleNames = [sup[i] + '_' + inf[i] for i in xrange(len(sup))]
    Nf = spineData.shape[0]
    res['spineAngles'] = {}
    res['spineAngles']['sagittal'] = {a: np.zeros((Nf,)) for a in spineAngleNames}
    
    # Create results directory if not existing
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
    
    # Process data
    for i in xrange(Nf):
        
        print('processing time frame %d ...' % i)        
        
        # Interpolate spline in sagittal plane
        spineDataSag = spineData[i,0:2,:].T # Np x 2
        Np = spineDataSag.shape[0]
        spineLineSag = spine.create2DSpline(spineDataSag)
        
        # Calculate slope of spine normal at the wanted points
        spineLineSagDer = spine.calcSplineTangentSlopes(spineDataSag, u='only_pts')
        normalSlopes = -spineLineSagDer[:,0] / spineLineSagDer[:,1]

        # Calculate angles between segments
        m1, m2 = normalSlopes[:-1], normalSlopes[1:]
        angles = spine.calcInterlinesAngle(m1, m2)
        for j in xrange(len(spineAngleNames)):
            res['spineAngles']['sagittal'][spineAngleNames[j]][i] = angles[j]
        
        if savePlots:
            
            # Create results directory if not existing
            figuresDir = os.path.join(resultsDir, 'figures')
            if not os.path.exists(figuresDir):
                os.mkdir(figuresDir)
            
            # Plot spine in 3D
#            fig = plt.figure()
#            ax = fig.gca(projection='3d')
#            ax.scatter(spineData[i,2,:], spineData[i,0,:], spineData[i,1,:])
#            mplh.set_axes_equal(ax)          
            
            # Plot spine in sagittal/frontal plane
            fig = plt.figure()
            
            plt.subplot(1, 2, 1)
            plt.plot(spineDataSag[:,0], spineDataSag[:,1], 'o')
            plt.plot(spineLineSag[:,0], spineLineSag[:,1], lw=3)
            ax = plt.gca()
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            xN = np.tile([-1000, 1000], [Np, 1])
            yN = (xN - spineDataSag[:,:1]) * normalSlopes[:,None] + spineDataSag[:,1:2]
            plt.plot(xN.T, yN.T, 'k')
            plt.axis('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.title('Sagittal')
            plt.xlabel('X pelvis (anterior +)')
            plt.ylabel('Y pelvis (up +)')
            
            plt.subplot(1, 2, 2)
            plt.plot(spineData[i,2,:], spineData[i,1,:], 'o')
            plt.axis('equal')
            plt.title('Frontal')
            plt.xlabel('Z pelvis (right +)')
            plt.savefig(os.path.join(figuresDir, 'tf_%04d.png' % i), format='png', orientation='landscape')
    
    # Create MATLAB-friendly reslts structure
    def adjustDictForML(d):
        dML = {}
        for k in d:
            kML = k.replace(' ', '_')
            dML[kML] = d[k]
        return dML
    resML = res.copy()
    resML['newMarkers'] = adjustDictForML(res['newMarkers'])        
    resML['spineAngles']['sagittal'] = adjustDictForML(res['spineAngles']['sagittal'])
    
    # Save data to MAT file
    fio.writeMATFile(os.path.join(resultsDir, 'results.mat'), resML)
        
    return res
    
    
    

    
    