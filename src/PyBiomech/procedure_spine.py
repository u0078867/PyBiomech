"""
.. module:: procedure_or
   :synopsis: helper module for procedures used with Oxford-Rig (IORT UZLeuven)

"""

import numpy as np

import fio, kine, spine

import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

   
    
    
    

def performSpineAnalysis(
                        markers1,
                        markers2,
                        singleMarkersDescFile,
                        segmentsDescFile,
                        clustersDescDir,
                        spinePointNames,
                        resultsDir,
                        anglesDef = {},
                        frames = None,
                        sagSpineSplineOrder=3,
                        froSpineSplineOrder=3,
                        savePlots=True,
                        ):
                            
    # calculate THORAX anatomical reference frame for acquisition system 1
    if set(['CLAV','STRN','C7','T9']) <= set(markers1.keys()):
        print('Creating thorax reference frame for acquisition system 1 with 4 markers')
        markers1Tho = {}
        markers1Tho['IJ'] = markers1['CLAV']
        markers1Tho['PX'] = markers1['STRN']
        markers1Tho['C7'] = markers1['C7']
        markers1Tho['T8'] = markers1['T9']
        RTho1, OTho1 = kine.thoraxPoseISB(markers1Tho)
        RTTho1 = kine.composeRotoTranslMatrix(RTho1, OTho1)
        RTTho1i = kine.inv2(RTTho1)
        isTho1Visible = True
        tho1Markers = ['CLAV','STRN','C7','T9']
    else:
        markers1Tho = {}
        for m in ['CLAV','STRN','C7','T9']:
            if m in markers1:
                markers1Tho[m] = markers1[m]
        if len(markers1Tho.keys()) == 3:
            print('Creating thorax reference frame for acquisition system 1 with 3 markers')
            RTho1, OTho1 = kine.markersClusterFun(markers1Tho, markers1Tho.keys())
            RTTho1 = kine.composeRotoTranslMatrix(RTho1, OTho1)
            RTTho1i = kine.inv2(RTTho1)
            isTho1Visible = True
            tho1Markers = markers1Tho.keys()
        else:
            print('Cannot create thorax reference frame for acquisition system 1')
            RTTho1i = None
            isTho1Visible = False
    
    # calculate PELVIS anatomical reference frame for acquisition system 1
    if set(['RASI','LASI','RPSI','LPSI']) <= set(markers1.keys()):
        print('Creating pelvis reference frame for acquisition system 1 with 4 markers')
        RPel1, OPel1 = kine.pelvisPoseISB(markers1, s='R')
        RTPel1 = kine.composeRotoTranslMatrix(RPel1, OPel1)
        RTPel1i = kine.inv2(RTPel1)
        isPel1Visible = True
        pel1Markers = ['RASI','LASI','RPSI','LPSI']
    else:
        markers1Pel = {}
        for m in ['RASI','LASI','RPSI','LPSI']:
            if m in markers1:
                markers1Pel[m] = markers1[m]
        if len(markers1Pel.keys()) == 3:
            print('Creating pelvis reference frame for acquisition system 1 with 3 markers')
            RPel1, OPel1 = kine.pelvisPoseNoOneASI(markers1Pel, s='R')
            RTPel1 = kine.composeRotoTranslMatrix(RPel1, OPel1)
            RTPel1i = kine.inv2(RTPel1)
            isPel1Visible = True
            pel1Markers = markers1Pel.keys()
        else:
            print('Cannot create pelvis reference frame for acquisition system 1')
            RTPel1i = None
            isPel1Visible = False
    
    # calculate THORAX anatomical reference frame for acquisition system 2
    if len(tho1Markers) == 4:
        markers2Tho = {}
        markers2Tho['IJ'] = markers2['CLAV']
        markers2Tho['PX'] = markers2['STRN']
        markers2Tho['C7'] = markers2['C7']
        markers2Tho['T8'] = markers2['T9']
        RTho2, OTho2 = kine.thoraxPoseISB(markers2Tho)
    else:
        RTho2, OTho2 = kine.markersClusterFun(markers2, tho1Markers)
    RTTho2 = kine.composeRotoTranslMatrix(RTho2, OTho2)
    
    # calculate PELVIS anatomical reference frame for acquisition system 2
    if len(pel1Markers) == 4:
        RPel2, OPel2 = kine.pelvisPoseISB(markers2, s='R')
    else:
        RPel2, OPel2 = kine.pelvisPoseNoOneASI(markers2, s='R')
    RTPel2 = kine.composeRotoTranslMatrix(RPel2, OPel2)
    RTPel2i = kine.inv2(RTPel2)
    
    # Handles single markers
    markers2New = markers2.copy()
    singleMarkersInfo = fio.readStringListMapFile(singleMarkersDescFile)
    for m in singleMarkersInfo:
        markerName = m
        segment = singleMarkersInfo[m][0]
        if len(singleMarkersInfo[m]) > 1:
            rawMarkerName = singleMarkersInfo[m][1]
        else:
            rawMarkerName = markerName
        
        # Express real marker in anatomical reference frame 
        if segment == 'thorax':
            RT1i = RTTho1i
        elif segment == 'pelvis':
            RT1i = RTPel1i
        else:
            raise Exception('segment must be one of: thorax, pelvis')
        
        if RT1i is None:
            print('single marker %s cannot be corrected: its local position in %s cannot be calculated, %s missing' % (markerName, segment, segment))
            continue
        
        targetMarkers = {}
        targetMarkers[markerName + ' base'] = markers1[rawMarkerName + ' base']
        targetMarkers['True ' + markerName] = markers1['True ' + rawMarkerName]
        targetMarkersLoc = kine.changeMarkersReferenceFrame(targetMarkers, RT1i)
        
        if segment == 'thorax':
            RT2 = RTTho2
        elif segment == 'pelvis':
            RT2 = RTPel2
        markers2New.update(kine.changeMarkersReferenceFrame(targetMarkersLoc, RT2))
    
    if isTho1Visible and isPel1Visible:
    
        # Handles clusters
        segmentsInfo = fio.readStringListMapFile(segmentsDescFile)
        
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
            
            # SVD for acquisition system 1
            args = {}
            markersLoc = clusterInfoSpec.copy()
            args['mkrsLoc'] = markersLoc
            args['verbose'] = True
            args['useOriginFromTrilat'] = False
            mkrList = markersLoc.keys()
            R1, T1, infoSVD1 = kine.rigidBodySVDFun2(markers1, mkrList, args)
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
            args['useOriginFromTrilat'] = False
            mkrList = markersLoc.keys()
            R2, T2, infoSVD2 = kine.rigidBodySVDFun2(markers2, mkrList, args)
            RT2 = kine.composeRotoTranslMatrix(R2, T2)
            
            # Copy and rename cluster base marker to be expressed in system 2 in global reference frame
            targetMarkersLoc[clusterBaseNameSpec + ' 2'] = clusterBaseMarkerLoc
            
            # Update list of markers for system 2 in global reference frame
            markers2New.update(kine.changeMarkersReferenceFrame(targetMarkersLoc, RT2))
    
    # Express spine points in pelvis reference frame
    spinePointNamesNew = spinePointNames[:]
    AIAPointNames = ['Apex top', 'Inflex', 'Apex bottom']
    spinePointNamesNewAll = spinePointNamesNew + AIAPointNames
    for i, m in enumerate(spinePointNames):
        # If wanted point name does not exist
        if m not in markers2New:
            # If wanted point is a True-type point
            if m.find('True') > -1:
                spinePointNamesNew[i] = m[5:]
    spinePoints = {m: markers2New[m] for m in spinePointNamesNew}
    spinePointsPel = kine.changeMarkersReferenceFrame(spinePoints, RTPel2i)
    
    # Create extra points of interest
    extraPoints = {}
    extraPoints['C7 for CVA'] = markers2New['True C7'].copy()
    extraPoints['C7 for CVA'][:,2] = markers2New['True SACR'][:,2]
    extraPointsPel = kine.changeMarkersReferenceFrame(extraPoints, RTPel2i)
    
    # Merge spine points in one array
    spineData = np.stack([spinePointsPel[m] for m in spinePointNamesNew], axis=2)  # Nf x 3 x Np
    extraData = np.stack([extraPointsPel[m] for m in ['C7 for CVA']], axis=2)  # Nf x 3 x Ne
    
    # Init results
    res = {}
    res['newMarkers'] = markers2New
    res['newMarkers'].update(extraPoints)
    sup, inf = spinePointNamesNew[:-1], spinePointNamesNew[1:]
    spineAngleNames = [sup[i] + '_' + inf[i] for i in xrange(len(sup))]
    Nf = spineData.shape[0]
    res['spineAngles'] = {}
    res['spineAngles']['sagittal'] = {a: np.zeros((Nf,)) for a in spineAngleNames}
    res['spineAngles']['frontal'] = {a: np.zeros((Nf,)) for a in spineAngleNames}
    customAngleNames = anglesDef.keys()
    for angleName in customAngleNames:
        space = anglesDef[angleName][0]
        res['spineAngles'][space][angleName] = np.zeros((Nf,))
    res['extraData'] = {}
    res['extraData']['SVA'] = np.zeros((Nf,))
    res['extraData']['SagApexTopHeight'] = np.zeros((Nf,))
    res['extraData']['SagInflexHeight'] = np.zeros((Nf,))
    res['extraData']['SagApexBottomHeight'] = np.zeros((Nf,))
    
    # Create results directory if not existing
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
        
    iRange = range(Nf)
    if frames is not None:
        iRange = frames
    
    # Process data
    for i in iRange:
        
        print('processing time frame %d ...' % i)        
        
        # Interpolate spline in sagittal plane
        #spineDataSag = spineData[i,0:2,:].T # Np x 2
        spineDataSag = spineData[i,1::-1,:].T # Np x 2
        extraDataSag = extraData[i,1::-1,:].T # Ne x 2
        Np = spineDataSag.shape[0]
        #spineLineSag = spine.create2DSpline(spineDataSag, order=sagSpineSplineOrder)
        spineLineSag = spine.create2DPolynomial(spineDataSag, order=sagSpineSplineOrder)
        
        # Calculate slope of spine normal at the wanted points
        #spineLineSagDer = spine.calcSplineTangentSlopes(spineDataSag, u='only_pts')
        spineLineSagDer = spine.calcPolynomialTangentSlopes(spineDataSag, u='only_pts', k=sagSpineSplineOrder)
        normalSlopesSag = -spineLineSagDer[:,1] / spineLineSagDer[:,0]
        normalInterceptsSag = spineDataSag[:,0] - normalSlopesSag * spineDataSag[:,1]
        
        # Search apex and inflexion points (AIA)
        uDense = np.arange(0, 1.001, 0.001)
        der1Dense = spine.calcPolynomialDerivatives(spineDataSag, u=uDense, k=sagSpineSplineOrder, der=1)[:,1]
        ndxDer1ChangeSign = np.append(np.diff(np.sign(der1Dense)), [False]) <> 0
        if ndxDer1ChangeSign.sum() <> 2:
            raise Exception('sagittal: there seems to be not exactly 2 apex points')
        der2Dense = spine.calcPolynomialDerivatives(spineDataSag, u=uDense, k=sagSpineSplineOrder, der=2)[:,1]
        ndxDer2ChangeSign = np.append(np.diff(np.sign(der2Dense)), [False]) <> 0
        win = np.cumsum(ndxDer1ChangeSign)
        win[win <> 1] = 0
        win = win.astype(np.bool)
        ndxDer2ChangeSign = ndxDer2ChangeSign & win
        if ndxDer2ChangeSign.sum() <> 1:
            raise Exception('sagittal: there seems to be not exactly 1 inflection point')
        ndxU = ndxDer1ChangeSign | ndxDer2ChangeSign
        spineDataAIASag = spine.evalPolynomial(spineDataSag, u=uDense[ndxU], k=sagSpineSplineOrder)
        res['extraData']['SagApexTopHeight'][i] = spineDataAIASag[0,0]
        res['extraData']['SagInflexHeight'][i] = spineDataAIASag[1,0]
        res['extraData']['SagApexBottomHeight'][i] = spineDataAIASag[2,0]
        spineDataSagAll = np.vstack((spineDataSag, spineDataAIASag))
        
        # Calculate slope of spine normal at the wanted points (AIA)
        spineLineAIASagDer = spine.calcPolynomialTangentSlopes(spineDataAIASag, u='only_pts', k=sagSpineSplineOrder)
        normalSlopesAIASag = -spineLineAIASagDer[:,1] / spineLineAIASagDer[:,0]
        normalInterceptsAIASag = spineDataAIASag[:,0] - normalSlopesAIASag * spineDataAIASag[:,1]
        
        normalSlopesSagAll = np.concatenate((normalSlopesSag, normalSlopesAIASag))
        normalInterceptsSagAll = np.concatenate((normalInterceptsSag, normalInterceptsAIASag))

        # Calculate angles between segments
        m1, m2 = normalSlopesSag[:-1], normalSlopesSag[1:]
        q1, q2 = normalInterceptsSag[:-1], normalInterceptsSag[1:]
        xCrossPoint = (q2 - q1) / (m1 - m2)
#        yCrossPoint = m1 * xCrossPoint + q1
        angleSign = (xCrossPoint > spineDataSag[:-1,1]) & (xCrossPoint > spineDataSag[1:,1])
        angleSign = 2 * (angleSign - 0.5)
        angles = angleSign * spine.calcInterlinesAngle(m1, m2)
        for j in xrange(len(spineAngleNames)):
            res['spineAngles']['sagittal'][spineAngleNames[j]][i] = angles[j]
            
        # Calculate SVA
        SVASign = extraDataSag[0,1] > spineDataSag[-1,1]
        SVASign = 2 * (SVASign - 0.5)
        res['extraData']['SVA'][i] = SVASign * np.linalg.norm(spineDataSag[-1,:] - extraDataSag[0,:])
            
        # Interpolate spline in frontal plane
#        spineDataFro = spineData[i,2:0:-1,:].T # Np x 2
        spineDataFro = spineData[i,1:,:].T # Np x 2
        Np = spineDataFro.shape[0]
#        spineLineFro = spine.create2DSpline(spineDataFro, order=froSpineSplineOrder)
        spineLineFro = spine.create2DPolynomial(spineDataFro, order=froSpineSplineOrder)
        
        # Calculate slope of spine normal at the wanted points
#        spineLineFroDer = spine.calcSplineTangentSlopes(spineDataFro, u='only_pts')
        spineLineFroDer = spine.calcPolynomialTangentSlopes(spineDataFro, u='only_pts', k=froSpineSplineOrder)
        normalSlopesFro = -spineLineFroDer[:,1] / spineLineFroDer[:,0]
        normalInterceptsFro = spineDataFro[:,0] - normalSlopesSag * spineDataFro[:,1]

        # Calculate angles between segments
        m1, m2 = normalSlopesFro[:-1], normalSlopesFro[1:]
        q1, q2 = normalInterceptsFro[:-1], normalInterceptsFro[1:]
        xCrossPoint = (q2 - q1) / (m1 - m2)
#        yCrossPoint = m1 * xCrossPoint + q1
        angleSign = (xCrossPoint > spineDataSag[:-1,1]) & (xCrossPoint > spineDataSag[1:,1])
        angleSign = 2 * (angleSign - 0.5)
        angles = angleSign * spine.calcInterlinesAngle(m1, m2)
        for j in xrange(len(spineAngleNames)):
            res['spineAngles']['frontal'][spineAngleNames[j]][i] = angles[j]
            
        # Calculate custom angles
        for angleName in customAngleNames:
            plane = anglesDef[angleName][0]
            p1 = anglesDef[angleName][1]
            p2 = anglesDef[angleName][2]
            if plane == 'sagittal':
                normalSlopes = normalSlopesSagAll
                normalIntercepts = normalInterceptsSagAll
            elif plane == 'frontal':
                normalSlopes = normalSlopesFro
                normalIntercepts = normalInterceptsFro
            try:
                i1 = spinePointNamesNewAll.index(p1)
                i2 = spinePointNamesNewAll.index(p2)
            except:
                raise Exception('%s and/or %s names are not recognized' % (p1,p2))
            m1, m2 = normalSlopes[i1], normalSlopes[i2]
            q1, q2 = normalIntercepts[i1], normalIntercepts[i2]
            xCrossPoint = (q2 - q1) / (m1 - m2)
#            yCrossPoint = m1 * xCrossPoint + q1
            angleSign = (xCrossPoint > spineDataSagAll[i1,1]) & (xCrossPoint > spineDataSagAll[i2,1])
            angleSign = 2 * (angleSign - 0.5)
            angle = angleSign * spine.calcInterlinesAngle(m1, m2)
            res['spineAngles'][plane][angleName][i] = angle
        
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
#            plt.show()
            
            # Plot spine in sagittal/frontal plane
            fig = plt.figure()
            
            plt.subplot(1, 2, 1)
            plt.plot(spineDataSag[:,1], spineDataSag[:,0], 'o')
            plt.plot(spineLineSag[:,1], spineLineSag[:,0], lw=3)
            plt.plot(spineDataAIASag[:,1], spineDataAIASag[:,0], 'rx')
            plt.plot(extraDataSag[0,1], extraDataSag[0,0], 'bo')
            ax = plt.gca()
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            xN = np.tile([-1000, 1000], [Np, 1])
            yN = (xN - spineDataSag[:,1][:,None]) * normalSlopesSag[:,None] + spineDataSag[:,0][:,None]
            plt.plot(xN.T, yN.T, 'k')
#            plt.plot(xCrossPoint, yCrossPoint, 'bo')
            plt.axis('equal')
            ax.set_xlim(xlim)
#            ax.set_xlim((-500,100))
            ax.set_ylim(ylim)
            plt.title('Sagittal')
            plt.xlabel('X pelvis (anterior +)')
            plt.ylabel('Y pelvis (up +)')
            
            plt.subplot(1, 2, 2)
#            plt.plot(spineData[i,2,:], spineData[i,1,:], 'o')
#            plt.axis('equal')
#            plt.title('Frontal')
#            plt.xlabel('Z pelvis (right +)')
            plt.plot(spineDataFro[:,1], spineDataFro[:,0], 'o')
            plt.plot(spineLineFro[:,1], spineLineFro[:,0], lw=3)
            ax = plt.gca()
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            xN = np.tile([-1000, 1000], [Np, 1])
            yN = (xN - spineDataFro[:,1][:,None]) * normalSlopesFro[:,None] + spineDataSag[:,0][:,None]
            plt.plot(xN.T, yN.T, 'k')
            plt.axis('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
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
    resML['spineAngles']['frontal'] = adjustDictForML(res['spineAngles']['frontal'])
    
    # Save data to MAT file
    fio.writeMATFile(os.path.join(resultsDir, 'results.mat'), resML)
        
    return res
    
    
    

    
    