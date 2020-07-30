"""
.. module:: kine
   :synopsis: helper module for kinematics

"""

import numpy as np
#from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from . import rotations as rot




def markersClusterFun(mkrs, mkrList):
    """Default function for calculating a roto-translation matrix from a cluster
    of markers to laboratory reference frame. It is based on the global position
    for the markers only, and there is not assumption of rigid body.
    The reference frame is defined as:

    - X versor from mkrList[-2] to mkrList[-1]
    - Z cross-product between X and versor from mkrList[-2] to mkrList[-3]
    - Y cross product between Z and X
    - Origin: mkrList[-2]

    Parameters
    ----------
    mkrs : dict
        Dictionary where each key is a maker name and each value
        is a N x 3 np.ndarray of 3D coordinates, where N is the number of time frames.

    mkrList : list
        List of marker names, whenever the names order is important.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    """

    # Define markers to use
    #M1 = mkrs[mkrList[-4]]
    M2 = mkrs[mkrList[-3]]
    M3 = mkrs[mkrList[-2]]
    M4 = mkrs[mkrList[-1]]

    # Create versors
    X = getVersor(M4 - M3)
    Z = getVersor(np.cross(X, M2 - M3))
    Y = getVersor(np.cross(Z, X))

    # Create rotation matrix from probe reference frame to laboratory reference frame
    R = np.array((X.T, Y.T, Z.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Create position vector
    T = M3.copy()

    # Return data
    return R, T


def changeMarkersReferenceFrame(mkrs, Rfull):
    """Express markers in another reference frame.

    Parameters
    ----------
    mkrs : dict
        Dictionary where each key is a maker name and each value
        is a N x 3 np.ndarray of 3D coordinates, where N is the number of time frames.

    Rfull : np.ndarray
        N x 4 x 4 affine matrix from current refence frame
        to new reference frame, for N frames.

    Returns
    -------
    dict
        Same structure as ``mkrs``, but with new coordinates.

    """

    # Calculate marker coordinates in local reference frame
    mkrsNew = {}
    mkrList = mkrs.keys()
    for m in mkrList:
        M = mkrs[m][:]  # copy
        if len(M.shape) == 1:
            N = Rfull.shape[0]
            M = np.tile(M, (N,1))
        M = np.hstack((M, np.ones((M.shape[0],1))))[:,:,None]
        mkrsNew[m] = dot3(Rfull, M)[:,0:3,0]
    return mkrsNew


def rigidBodySVDFun(mkrs, mkrList, args):
    """Function for calculating the optimal roto-translation matrix from a rigid
    cluster of markers to laboratory reference frame. The computation, by using
    SVD, minimizes the RMSE between the markers in the laboratory reference frame
    and the position of the markers in the local reference frame.
    See ``rigidBodyTransformation()`` for more details.

    Parameters
    ----------
    mkrs : dict
        Dictionary where each key is a marker name and each value
        is a N x 3 np.ndarray of 3D coordinates, where N is the number of time frames.

    mkrList : list
        List of marker names used in the SVD.

    args : mixed
        Additional arguments:
        - 'mkrsLoc': dictionary where keys are marker names and values are 3-elem
        np.arrays indicating the coordinates in the local reference frame.
        - 'verbose': boolean indicating verbosity of printing messages.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    """
    
    # Get verbosity
    if 'verbose' in args:
        verbose = args['verbose']
    else:
        verbose = True

    # Extract coordinates of markers in rigid local reference frame
    mkrsLoc = args['mkrsLoc']

    # Create Nmarkers x 3 matrix for local coordinates
    x = np.array([np.asarray(mkrsLoc[m]) for m in mkrList])

    # Loop for each time frame
    Nf = mkrs[mkrList[0]].shape[0]
    R = np.zeros((Nf, 3, 3))
    T = np.zeros((Nf, 3))
    RMSE = np.zeros((Nf,))
    eMax = np.zeros((Nf,))
    eMaxMarker = Nf * [None]
    for i in range(0,Nf):

        # Create Nmarkers x 3 matrix for global coordinates
        y = np.array([mkrs[m][i,:].tolist() for m in mkrList])

        # Calculate number of visible markers
        idxNan = np.isnan([y[:,0]])[0]
        Nv = y.shape[0] - idxNan.sum()

        # Check minimum markers number
        if Nv >= 3:

            # Calculate optimal roto-translation matrix
            Ri, Ti, ei = rigidBodyTransformation(x[~idxNan,:], y[~idxNan,:])

        else:

            # Set data to nan
            Ri = np.empty((3,3)) * np.nan
            Ti = np.empty((3,)) * np.nan
            ei = np.empty((y.shape[0],)) * np.nan
            print('Only %d markers are visible for frame %d. Data will be set to nan' % (Nv, i))

        # Calculate RSME
        RMSEi = np.sqrt(np.sum(ei))
        iMaxi = np.argmax(ei)
        eMaxi = np.max(ei)

        if verbose:
            print('RMSE for rigid pose estimation (%d markers) for frame %d: %.5f mm. Max distance for %s: %.5f mm' % (Nv, i, RMSEi, mkrList[iMaxi], eMaxi))
        
        # Insert data
        R[i,:,:] = Ri
        T[i,:] = Ti
        RMSE[i] = RMSEi
        eMax[i] = eMaxi
        eMaxMarker[i] = mkrList[iMaxi]

    # Return data
    info = {}
    info['RMSE'] = RMSE
    info['eMax'] = eMax
    info['eMaxMarker'] = eMaxMarker
    return R, T, info
    
    
def rigidBodySVDFun2(mkrs, mkrList, args):
    """Function for calculating the optimal roto-translation matrix from a rigid
    cluster of markers to laboratory reference frame. The computation, by using
    SVD, minimizes the RMSE between the markers in the laboratory reference frame
    and the position of the markers in the local reference frame. Additionally,
    if requested, the origin marker is used as additional marker: its local
    position in [0,0,0], while its laboratory position is created by trilaterating
    3 markers. It can only be applied when the list of markers is of length 3.
    See ``rigidBodyTransformation()`` for more details.

    Parameters
    ----------
    mkrs : dict
        Dictionary where each key is a marker name and each value
        is a N x 3 np.ndarray of 3D coordinates, where N is the number of time frames.

    mkrList : list
        List of marker names used in the SVD.

    args : mixed
        Additional arguments:
        - 'mkrsLoc': dictionary where keys are marker names and values are 3-elem
        np.arrays indicating the coordinates in the local reference frame.
        - 'verbose': boolean indicating verbosity of printing messages.
        - 'useOriginFromTrilat': create a 4th marker to be used by SVD.
        If ``len(mkrList) <> 3``, this option is ignored.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    """
    
    # Get verbosity
    if 'verbose' in args:
        verbose = args['verbose']
    else:
        verbose = True
        
    # Get request for additional origin from trilat
    if 'useOriginFromTrilat' in args:
        useOriginFromTrilat = args['useOriginFromTrilat']
    else:
        useOriginFromTrilat = False

    # Extract coordinates of markers in rigid local reference frame
    mkrsLoc = args['mkrsLoc']

    # Create Nmarkers x 3 matrix for local coordinates
    mkrList_ = mkrList[:]
    x = np.array([np.asarray(mkrsLoc[m]) for m in mkrList])
    
    # Check if trilat can actually be used
    useOriginFromTrilat = useOriginFromTrilat and len(mkrList) == 3
    
    if useOriginFromTrilat:
        x = np.vstack([x, [0,0,0]])
        mkrList_.append(','.join(mkrList) + ' center')

    # Loop for each time frame
    Nf = mkrs[mkrList[0]].shape[0]
    R = np.zeros((Nf, 3, 3))
    T = np.zeros((Nf, 3))
    RMSE = np.zeros((Nf,))
    eMax = np.zeros((Nf,))
    eMaxMarker = Nf * [None]
    for i in range(0,Nf):
        # Create Nmarkers x 3 matrix for global coordinates
        y = np.array([mkrs[m][i,:].tolist() for m in mkrList])
        
        if useOriginFromTrilat:
            p = np.array(y)
            r = [np.sqrt(np.sum(mkrsLoc[m]**2)) for m in mkrList]
            u, v = trilateration(p, r)
            c = p[0,:] + u + v
            y = np.vstack([y, c])

        # Calculate number of visible markers
        idxNan = np.isnan([y[:,0]])[0]
        Nv = y.shape[0] - idxNan.sum()

        # Check minimum markers number
        if Nv >= 3:

            # Calculate optimal roto-translation matrix
            Ri, Ti, ei = rigidBodyTransformation(x[~idxNan,:], y[~idxNan,:])

        else:

            # Set data to nan
            Ri = np.empty((3,3)) * np.nan
            Ti = np.empty((3,)) * np.nan
            ei = np.empty((y.shape[0],)) * np.nan
            print('Only %d markers are visible for frame %d. Data will be set to nan' % (Nv, i))

        # Calculate RSME
        RMSEi = np.sqrt(np.sum(ei))
        iMaxi = np.argmax(ei)
        eMaxi = np.max(ei)

        if verbose:
            print('RMSE for rigid pose estimation (%d markers) for frame %d: %.5f mm. Max distance for %s: %.5f mm' % (Nv, i, RMSEi, mkrList_[iMaxi], eMaxi))
        
        # Insert data
        R[i,:,:] = Ri
        T[i,:] = Ti
        RMSE[i] = RMSEi
        eMax[i] = eMaxi
        eMaxMarker[i] = mkrList_[iMaxi]

    # Return data
    info = {}
    info['RMSE'] = RMSE
    info['eMax'] = eMax
    info['eMaxMarker'] = eMaxMarker
    return R, T, info


def rigidBodyTransformation(x, y):
    """Estimate or rigid rotation and translation between x and y in such a way
    that y = Rx + t + e is optimal in a least square optimal. Details of the
    algorithm can be found here:

    - Arun et al. (1987)
    - Woltring (1992)
    - Soderkvist & Wedin (1993)
    - Challis (1995)

    Parameters
    ----------
    x : np.ndarray
        Nm x 3 array containing coordinates for Nm points in
        the local rigid reference frame.

    y : np.ndarray
        Nm x 3 array containing coordinates for Nm points in
        the global reference frame.

    Returns
    -------
    R : np.ndarray
        3 x 3 estimated rotation matrix.

    t : np.ndarray
        3-elem translation t.

    e : np.ndarray
        Nm-elem estimated error e.

    """

    # Get markers number
    Nmarkers = x.shape[0]

    # Calculation of the cross-dispersion matrix C
    xmean = np.mean(x, axis=0)
    ymean = np.mean(y, axis=0)
    A = x - np.dot(np.ones((Nmarkers,3)), np.diag(xmean))
    B = y - np.dot(np.ones((Nmarkers,3)), np.diag(ymean))
    C = np.dot(B.T, A) / Nmarkers

    # Singular value decomposition of C
    U, S, V = np.linalg.svd(C)
    tol = 0.00002
    s = S#np.diag(S)
    Srank = np.sum(s > tol)
    if Srank < 2:
        raise Exception('Markers are probably colinear aligned')
#    if Srank < 3:
#        # All markers in one plane
#        # Calculate cross product, i.e. normal vector
#        U[:,2] = np.cross(U[:,0], U[:,1])
#        V[:,2] = np.cross(V[:,0], V[:,1])

    # Calculation of R, t and e
    D = np.round(np.linalg.det(np.dot(U, V.T))) # if D=-1 correction is needed:
    R = np.dot(np.dot(U, np.diag([1,1,D])), V)
    t = (ymean.T - np.dot(R, xmean).T).T # vectors are internal in the [x y z] format
    #t=t'; # patch to accomodate external format

    # Calculate estimation error
    yestimated = np.dot(x, R.T) + np.dot(np.ones((Nmarkers,3)), np.diag(t)) # vectors are internal in the [x y z] format
    dy = yestimated - y
    e = np.sqrt(np.sum(dy**2, axis=1)).squeeze()

    return R, t, e


def pca(D):
    """Run Principal Component Analysis on data matrix. It performs SVD
    decomposition on data covariance matrix.

    Parameters
    ----------
    D : np.ndarray
        Nv x No matrix, where Nv is the number of variables
        and No the number of observations.

    Returns
    -------
    list
        U, s as out of SVD (``see np.linalg.svd``)

    """
    cov = np.cov(D)
    U, s, V = np.linalg.svd(cov)
    return U, s


def dot2(a, b):
    """Compute K matrix products between a M x N array and a K x N x P
    array in a vectorized way.

    Parameters
    ----------
    a : np.ndarray
        M x N array

    b : np.ndarrayK x N x P array
        np.ndarray

    Returns
    -------
    np.ndarray
        K x M x P array

    """

    return np.transpose(np.dot(np.transpose(b,(0,2,1)),a.T),(0,2,1))


def vdot2(a, b):
    """Compute dot product in a vectorized way.

    Parameters
    ----------
    a : np.ndarray
        K x 3 array

    b : np.ndarray
        K x 3 array

    Returns
    -------
    np.ndarray
        K-elems array

    """
    r = np.sum(a * b, axis=1)
    return r


def dot3(a, b):
    """Compute K matrix products between a K x M x N array and K x N x P
    array in a vectorized way.

    Parameters
    ----------
    a : np.ndarray
        K x M x N array

    b : np.ndarray
        K x N x P array

    Returns
    -------
    np.ndarray
        K x M x P array

    """

    return np.einsum('kmn,knp->kmp', a, b)


def getVersor(a):
    """Calculate versors of an array.

    Parameters
    ----------
    a : np.ndarray
        N x 3 array

    Returns
    -------
    np.ndarray
        N x 3 array of versors coordinates

    """

    #norm = np.sqrt(np.sum(np.multiply(np.mat(a),np.mat(a)),axis=1))
    norm = np.sqrt(np.sum(a**2, axis=1))
    r = a / (norm[:,None])
    return r


def interpSignals(x, xNew, D, kSpline=1):
    """Interpolate data array, with extrapolation. Data can contain NaNs.
    The gaps will not be filled.

    Parameters
    ----------
    D : np.ndarray
        N x M data array to interpolate (interpolation is column-wise).

    x : np.ndarray
        axis of the original data, with length N.

    xNew : np.ndarray
        New axis for the interpolation, with length P.

    kSpline : mixed
        See ``k`` in ``scipy.interpolate.InterpolatedUnivariateSpline()``.

    Returns
    -------
    np.ndarray
        P x M interpolated array

    """

    R = np.zeros((xNew.shape[0],D.shape[1])) * np.nan
    for i in range(0, D.shape[1]):
        idx = ~np.isnan(D[:,i])
        fIdx = InterpolatedUnivariateSpline(x, idx, k=1)
        f = InterpolatedUnivariateSpline(x[idx], D[idx,i], k=kSpline)
        #idxNew = np.round(fIdx(xNew)).astype(np.bool)
        idxNew = fIdx(xNew) >= 0.9
        R[idxNew,i] = f(xNew[idxNew])
    return R


def resampleMarker(M, x=None, origFreq=None, origX=None, step=None):
    """Resample marker data.
    The function first tries to see if the new time scale ``x`` and ``origFreq``
    (to create the old scale) or ``origX`` are available. If not, the
    resampling will take a frame each ``step`` frames.

    Parameters
    ----------
    M : np.ndarray
        N x 3 marker data array to resample

    x : np.ndarray
        The new time scale (in *s*) on which to peform the resampling.

    origFreq : double
        Frequency (in *Hz*) to recreate the old time scale.

    origX : np.ndarray
        The old time scale (in *s*).

    step : int
        Number of frames to skip when performing resampling not based on ``x``.

    Returns
    -------
    Mout : np.ndarray
        M x 3 resampled marker data

    ind : np.ndarray
        Indices of ``x`` intersecting time vector of the original ``M``.

    """
    if x is not None and (origFreq is not None or origX is not None):
        if origFreq is not None:
            N = M.shape[0]
            dt = 1. / origFreq
            x1 = np.linspace(0, (N-1)*dt, num=N)
        else:
            x1 = origX.copy()
        x2 = x.copy()
#        f = interp1d(x1, M, axis=0)
#        M2 = f(x2)
        M2 = interpSignals(x1, x2, M)
    elif step is not None:
        N = M.shape[0]
        x1 = np.linspace(0, N-1, num=N)
        x2 = np.arange(0, N-1, step)
#        f = interp1d(x1, M, axis=0)
#        M2 = f(x2)
        M2 = interpSignals(x1, x2, M)
    else:
        raise Exception('Impossible to resample')
    ind = np.nonzero((x2 >= x1[0]) & (x2 <= x1[-1]))[0]
    return M2, ind



def resampleMarkers(M, **kwargs):
    """Resample markers data.

    Parameters
    ----------
    M : dict
        Dictionary where keys are markers names and values are np.ndarray
        N x 3 marker data array to resample.

    **kwargs
        See ``resampleMarker()``.

    Returns
    -------
    resM : dict
        Resampled marker data

    ind : np.ndarray
        See ``resampleMarker()``.

    """
    resM = {}
    for m in M:
        resM[m], ind = resampleMarker(M[m], **kwargs)
    return resM, ind



def composeRotoTranslMatrix(R, T):
    """Create affine roto-translation matrix from rotation matrix and translation vector.

    Parameters
    ----------
    R : np.ndarray
        N x 3 x 3 rotation matrix, for N frames.

    T : np.ndarray
        N x 3 translation vector, for N frames.

    Returns
    -------
    np.ndarray
        N x 4 x 4 affine matrix.

    """
    Nf = R.shape[0]
    Rfull = np.concatenate((R, np.reshape(T,(Nf,3,1))), axis=2)
    b = np.tile(np.array((0,0,0,1)), (Nf,1))
    Rfull = np.concatenate((Rfull, np.reshape(b,(Nf,1,4))), axis=1)
    return Rfull


def decomposeRotoTranslMatrix(Rfull):
    """Extract rotation matrix and translation vector from affine roto-translation matrix.

    Parameters
    ----------
    Rfull : np.ndarray
        N x 4 x 4 affine matrix, for N frames.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix and second

    T : np.ndarray
        N x 3 translation vector, for N frames.

    """

    R = Rfull[:,0:3,0:3]
    T = Rfull[:,0:3,3]
    return R, T


def inv2(R):
    """Behaves like np.linalg.inv for multiple matrices, but does not raise
    exceptions if a matrix contains nans and it is not invertible.

    Parameters
    ----------
    R : np.ndarray
        N x M x M series of matrices to invert.

    Returns
    -------
    np.ndarray
        N x M x M array.

    """

    Rinv = np.zeros(R.shape) * np.nan
    idx = np.delete(np.arange(R.shape[0]), np.nonzero(np.isnan(R))[0])
    Rinv[idx,:,:] = np.linalg.inv(R[idx,:,:])
    return Rinv


def createClusterTemplate(markers, mkrList, timeWin='all_no_nan'):
    """Create cluster template data from existing markers data.

    Parameters
    ----------
    markers : dict
        Dictionary of point 3D coordinates. Keys are points names
        values are np.ndarray N x 3 representing 3D coordinates in the global
        reference frame, where N is the number of time frames.

    mkrList : list
        List of marker names to be used for the template.

    timeWin : mixed
        Represents which time frames to select for template creation.
        If str, it can be:

        - 'all_no_nan': all time frames apart from those where marker data is nan.
          If list, it must contain two values containing first and last frame for the
          time window to search into. Only non-nans will be used.
          If single value, it indicates the frame to use.

    Returns
    -------
    dict
        Dictionary where keys are marker names and values are 3-elem
        np.arrays indicating the coordinates in the cluster reference frame.

    """

    # Calculate roto-translation-matrix from local to laboratory reference frame
    R, T = markersClusterFun(markers, mkrList)

    # Invert roto-translation matrix
    Rfull = composeRotoTranslMatrix(R, T)
    Rfull = inv2(Rfull)

    # Express markers in the local rigid probe reference frame
    markersLoc = changeMarkersReferenceFrame(markers, Rfull)

    # Calculate reference frame with SVD
    if timeWin == 'all_no_nan':
        markersLoc = {m: np.nanmean(markersLoc[m], axis=0) for m in mkrList}
    elif isinstance(timeWin, (list,tuple)):
        i1 = timeWin[0]
        i2 = timeWin[1]
        markersLoc = {m: np.nanmean(markersLoc[m][i1:i2,:], axis=0) for m in mkrList}
    elif isinstance(timeWin, (int,float)):
        markersLoc = {m: markersLoc[m][timeWin,:] for m in mkrList}

    return markersLoc


def collinearNPointsStylusFun(P, args):
    """Tip reconstruction function for M collinear points.

    Parameters
    ----------
    P : dict
        Dictionary of point 3D coordinates. Keys are points names
        values are np.ndarray N x 3 representing 3D coordinates (in *mm*)
        in the global reference frame, where N is the number of time frames.

    args : dict
        Dictionary with the floowing keys:

        - 'markers': list of marker names to be used.
        - 'dist': dictionary of distances between points and tip. Keys must be
          present in the 'markers' list, values are distances (in *mm*).

    Returns
    -------
    np.ndarray
        N x 3 array representing 3D coordinates of the reconstructed tip
        (in *mm*).

    """

    # Get distances from stylus tip to each marker
    dist = args['dist']

    # Get markers
    markers = args['markers']

    # Difference between extreme markers
    existingMarkerIdxs = []
    for i in range(0, len(markers)):
        if markers[i] in P:
            existingMarkerIdxs.append(i)

    if len(existingMarkerIdxs) < 2:
        raise Exception('At least 2 collinear pointer markers must be visible')

    m1 = existingMarkerIdxs[0]
    m2 = existingMarkerIdxs[-1]
    E = P[markers[m1]] - P[markers[m2]]
    Nf = P[markers[m1]].shape[0]

    # Normalize distance
    N = E / np.linalg.norm(E, axis=1)[:,None]

    # Calculate N tips, with each distance
    X = np.zeros((Nf,0))
    Y = np.zeros((Nf,0))
    Z = np.zeros((Nf,0))
    for i in range(0, len(markers)):
        if i not in existingMarkerIdxs:
            continue
        tip = P[markers[i]] + dist[i] * N
        X = np.hstack((X, tip[:,0:1]))
        Y = np.hstack((Y, tip[:,1:2]))
        Z = np.hstack((Z, tip[:,2:3]))

    # Average the tips
    X = np.nanmean(X, axis=1)[:,None]
    Y = np.nanmean(Y, axis=1)[:,None]
    Z = np.nanmean(Z, axis=1)[:,None]
    tip = np.hstack((X,Y,Z))

    return tip
    
    
def nonCollinear5PointsStylusFun(P, args, verbose=True):
    """Tip reconstruction function for 5 non-collinear points.
    The five markers are supposed to be on the same plane.
    It performs tip reconstruction in differente phases:
    - stylus markers adjusting based on a template and SVD;
    - trilateration by 2 couples of 3 markers, giving in-plane vector "u" and
    off-plane vector "v" (see ``trilateration()``);
    - replace "v" position of the tip from trilateration by finding the versor 
    normal to least-square plane through the 5 markers, and multiplying it by
    a known distance;
    - average the results of the 2 computed tip positions.

    Parameters
    ----------
    P : dict
        Dictionary of point 3D coordinates. Keys are points names
        values are np.ndarray N x 3 representing 3D coordinates (in *mm*)
        in the global reference frame, where N is the number of time frames.

    args : dict
        Dictionary with the following necessary keys:

        - 'markers': list of marker names to be used.
        - 'pos': dictionary of position of markers in the local reference frame.
        Keys must be present in the 'markers' list, values are coordinates in
        a list. It is used for the SVD algorithm.
        - 'dist': dictionary of distances between points and tip. Keys must be
        present in the 'markers' list, values are distances (in *mm*).
        It is used for the trilateration algorithm, and they can be considered
        as the radii of the sphere reconstructed around the markers.
        - 'offPlaneDist': off-plane distance (in *m*, see ``trilateration()``).
        It is used for the trilateration algorithm.
        
    verbose : bool
        Indicates if to be verbose in printing messages.
         

    Returns
    -------
    np.ndarray
        N x 3 array representing 3D coordinates of the reconstructed tip
        (in *mm*).

    """        
    
    # Assign convenient names to markers
    mkrNames = args['markers']
    nameP1 = mkrNames[0]
    nameP2 = mkrNames[1]
    nameP3 = mkrNames[2]
    nameP4 = mkrNames[3]
    nameP5 = mkrNames[4]
    
    # Prepare common data for SVD
    dataSVD = {}
    dataSVD['mkrsLoc'] = args['pos']
    dataSVD['verbose'] = verbose
    
    if args['algoSVD'] == 1:
    
        # Perform SVD for triangle of markers P1-P3-P5
        print('Performing SVD with P1-P3-P5 ...')
        mkrNames1 = [nameP1, nameP3, nameP5]
        R1, T1, info1 = rigidBodySVDFun(P, mkrNames1, dataSVD)
        RT1 = composeRotoTranslMatrix(R1, T1)
        Pf1 = changeMarkersReferenceFrame(args['pos'], RT1)
        
        # Perform SVD for triangle of markers P2-P4-P5
        print('Performing SVD with P2-P4-P5 ...')
        mkrNames2 = [nameP2, nameP4, nameP5]
        R2, T2, info2 = rigidBodySVDFun(P, mkrNames2, dataSVD)
        RT2 = composeRotoTranslMatrix(R2, T2)
        Pf2 = changeMarkersReferenceFrame(args['pos'], RT2)
        
        # Get markers and distances for trilaterations
        P1 = Pf1[nameP1]
        P2 = Pf2[nameP2]
        P3 = Pf1[nameP3]
        P4 = Pf2[nameP4]
        P5 = .5 * (Pf1[nameP5] + Pf2[nameP5])
    
    elif args['algoSVD'] == 2:
        
        # Perform SVD with all visible points
        print('Performing SVD ...')
        R, T, info = rigidBodySVDFun(P, mkrNames, dataSVD)
#        try:
#            R, T, info = rigidBodySVDFun(P, ['Wand:WN','Wand:WE','Wand:WM','Wand:WS'], dataSVD)
#        except:
#            R, T, info = rigidBodySVDFun(P, ['WN','WE','WM','WS'], dataSVD)
        RT = composeRotoTranslMatrix(R, T)
        Pf = changeMarkersReferenceFrame(args['pos'], RT)
        
        # Get markers and distances for trilaterations
        P1 = Pf[nameP1]
        P2 = Pf[nameP2]
        P3 = Pf[nameP3]
        P4 = Pf[nameP4]
        P5 = Pf[nameP5]
        
    if args['algoTrilat'] == 0:
        
        tip = Pf['Tip']
        
    elif args['algoTrilat'] == 1:
                
        r = args['dist']
        r1 = r[nameP1]
        r2 = r[nameP2]
        r3 = r[nameP3]
        r4 = r[nameP4]
        r5 = r[nameP5]
    
        # Perform trilaterations
        print('Performing trilaterations ...')
        Nf = P1.shape[0]
        u1 = np.zeros((Nf,3))
        u2 = np.zeros((Nf,3))
        v = np.zeros((Nf,3))
        for i in range(Nf):
            
            # Perform trilateration with markers P1-P3-P5
            _u1, _v1 = trilateration(np.array([P1[i,:], P3[i,:], P5[i,:]]), np.array([r1, r3, r5]))
            #print np.linalg.norm(_v1)
            u1[i,:] = _u1
            
            # Perform trilateration with markers P1-P3-P5
            _u2, _v2 = trilateration(np.array([P2[i,:], P4[i,:], P5[i,:]]), np.array([r2, r4, r5]))
            #print np.linalg.norm(_v2)
            u2[i,:] = _u2
            
            # Find off-plane versor
            cplane = getNormalToLSPlane(np.array((P1[i,:], P2[i,:], P3[i,:], P4[i,:], P5[i,:])))
            _v = np.dot(np.cross(P3[i,:] - P2[i,:], P5[i,:] - P2[i,:]), cplane)
            v[i,:] = np.sign(_v) * args['offPlaneDist'] * cplane
        
        # Find the 2 tips
        print('Reconstructing 2 tips ...')
#        tip1 = P1 + u1 + v
#        tip2 = P2 + u2 + v
        tip1 = P1 + u1 + _v1
        tip2 = P2 + u2 + _v2
        
        # Take the average of the two estimations for the tip position
        print('Averaging 2 tips ...')
        tip = .5 * (tip1 + tip2)
    
    print('Tip calculated')
    return tip
    

def trilateration(p, r):
    """Perform trilateration algorithm to find the position of a point given
    its distance from 3 points and the position of these 3 points in a 
    common reference frame.
    In particular, this function seeks the intersecting point of 3 spheres
    defined by their centerpoint and radius. The intersection points can be 
    calculated as i1 = p1+u1+v and i2 = p1+u1-v. This problem can have two, 
    one or no solution; depending of the vector v (real, zero or non-existent).
    In the case of no solution, i = p1+u1 can be taken as an estimate of the 
    solution.

    Parameters
    ----------
    p : np.ndarray
        3 x 3 representing coordinates of the spheres centers, each row 
        being a point.
        
    r : np.ndarray
        Radii of the 3 spheres.

    Returns
    -------
    u1 : np.ndarray
        3-elem array representing position of the intersecting point of three 
        planes (the plane of the intersecting circle of sphere 1 and 2, 
        the plane of the intersecting circle of sphere 1 and 3, the plane 
        containing the three center points) relative to the point c1.
        In the case of no solution, this point can be taken as an estimate of
        the solution.
        
    v : np.ndarray
        Vector stretching from point u1 (in the plane of the three center
        points) to the two intersecting points.
        If v = [0,0,0], there is only one solution. 
        In the case of no solution, v = [np.NaN, np.NaN, np.NaN].

    """
    # Calculate vectors u1 and v
    p21 = p[1,:] - p[0,:]
    p31 = p[2,:] - p[0,:]
    c = np.cross(p21, p31)
    u1 = np.cross( ((np.linalg.norm(p21)**2 + r[0]**2 - r[1]**2) * p31 - (np.linalg.norm(p31)**2 + r[0]**2 - r[2]**2) * p21) / 2., c ) / np.linalg.norm(c)**2
    if r[0] >= np.linalg.norm(u1):
        v = np.sqrt(r[0]**2 - np.linalg.norm(u1)**2) * c / np.linalg.norm(c)
    else:
        v = np.ones((3,)) * np.nan
    return u1, v
    

def getNormalToLSPlane(P):
    """Get normal versor of the plane best fitting N points in the least-square
    sense.

    Parameters
    ----------
    P : np.ndarray
        N x 3 coordinates of the points.

    Returns
    -------
    np.ndarray
        3-elem array representing 3D coordinates of the plane versor.

    """
    N = P.shape[0]
    B = np.ones((N, 1))
    #n = (P.T*P)\(P.T*B)
    n = np.linalg.solve(np.dot(P.T, P), np.dot(P.T, B))
    n = n / np.linalg.norm(n)
    return n.squeeze()
        


class Stylus:
    """Helper class for reconstructing stylus tip using source points rigidly connected to stylus.
    """

    def __init__(self, P=None, fun=None, args=None):
        """Constructor
        """
        self.P = P
        self.tipFun = fun
        self.tipFunArgs = args


    def setPointsData(self, P):
        """Set source points 3D coordinates.

        Parameters
        ----------
        P : dict
            Dictionary of point 3D coordinates. Keys are points names,
            values are np.ndarray N x 3, where N is the number of time frames.

        """

        self.P = P


    def setTipReconstructionFunction(self, fun):
        """Set the function for tip reconstruction from source points.

        Parameters
        ----------
        fun : fun
            Function taking as input arguments ``P`` and, if not None, ``args``.
            It must return a N x 3 np.ndarray representing 3D coordinates of
            the reconstructed tip.

        """
        self.tipFun = fun


    def setTipReconstructionFunctionArgs(self, args):
        """Set additional arguments for tip reconstruction function.

        Parameters
        ----------
        args : mixed
            Argument passed to ``fun``.

        """
        self.tipFunArgs = args


    def reconstructTip(self):
        """Perform tip reconstruction.
        """

        if self.tipFunArgs == None:
            self.tip = self.tipFun(self.P)
        else:
            self.tip = self.tipFun(self.P, self.tipFunArgs)


    def getTipData(self):
        """Get tipa data.

        Returns
        -------
        np.ndarray
            N x 3 array representing 3D coordinates of the reconstructed tip.

        """
        return self.tip



def calculateStylusTipInCluster(stylus, markers, clusterMkrList, clusterArgs):
    """Helper function for:
    - markers cluster pose estimation (by SVD)
    - reconstruction of the stylus tip in the cluster reference frame.

    Parameters
    ----------
    markers : dict
        See ``mkrs`` in ``rigidBodySVDFun()``.

    clusterMkrList : list
        See ``mkrList`` in ``rigidBodySVDFun()``.

    clusterArgs : mixed
        See ``args`` in ``rigidBodySVDFun()``.

    Returns
    -------
    np.ndarray
        N x 3 array representing 3D coordinates of the reconstructed tip
        (in *mm*) in the cluster reference frame, where N is the number of
        time frames.

    """

    # Calculate reference frame
    R, T, info = rigidBodySVDFun(markers, clusterMkrList, args=clusterArgs)

    # Invert roto-translation matrix
    gRl = composeRotoTranslMatrix(R, T)
    lRg = inv2(gRl)

    # Reconstruct stylus tip
    stylus.setPointsData(markers)
    stylus.reconstructTip()
    tip = stylus.getTipData()

    # Average on the time frames
    tip = np.nanmean(tip, axis=0)

    # Add tip to available markers
    markers['Tip'] = tip

    # Express tip in the local rigid cluster reference frame
    tipLoc = changeMarkersReferenceFrame(markers, lRg)['Tip']

    return tipLoc
    
    
    
def thoraxPoseISB(mkrs):
    """Calculate roto-translation matrix from thorax (ISB conventions) to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'IJ': incisura jugularis
        - 'PX': processus xiphoideus,
        - 'C7': processus spinosus of the 7th cervical vertebra
        - 'T8': processus spinosus of the 8th thoracic vertebra

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Wu G, van der Helm FC, Veeger HE, Makhsous M, Van Roy P, Anglin C, Nagels J,
    Karduna AR, McQuade K, Wang X, Werner FW, Buchholz B; International Society of
    Biomechanics. ISB recommendation on definitions of joint coordinate systems of
    various joints for the reporting of human joint motion--Part II: shoulder, elbow,
    wrist and hand. J Biomech. 2005 May;38(5):981-992. Review. PubMed PMID: 15844264.


    """

    # Define markers to use
    IJ = mkrs['IJ']
    PX = mkrs['PX']
    C7 = mkrs['C7']
    T8 = mkrs['T8']

    # Create versors
    Otho = IJ.copy()
    Ytho = getVersor(0.5 * (IJ + C7) - 0.5 * (T8 + PX))
    Ztho = getVersor(np.cross(IJ - 0.5 * (T8 + PX), C7 - 0.5 * (T8 + PX)))
    Xtho = getVersor(np.cross(Ytho, Ztho))

    # Create rotation matrix from shank reference frame to laboratory reference frame
    R = np.array((Xtho.T, Ytho.T, Ztho.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Otho
    
    

def pelvisPoseISB(mkrs, s='R'):
    """Calculate roto-translation matrix from pelvis (ISB conventions) to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'RASI': right spina iliaca anterior superior
        - 'LASI': right spina iliaca anterior superior
        - 'RPSI': right spina iliaca posterior superior
        - 'LPSI': right spina iliaca posterior superior

    s : {'R', 'L'}
        Anatomical side.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Wu G, Siegler S, Allard P, Kirtley C, Leardini A, Rosenbaum D, Whittle M,
    D'Lima DD, Cristofolini L, Witte H, Schmid O, Stokes I; Standardization and
    Terminology Committee of the International Society of Biomechanics. ISB
    recommendation on definitions of joint coordinate system of various joints for
    the reporting of human joint motion--part I: ankle, hip, and spine. International
    Society of Biomechanics. J Biomech. 2002 Apr;35(4):543-8. PubMed PMID: 11934426.

    """

    # Define markers to use
    RASI = mkrs['RASI']
    LASI = mkrs['LASI']
    RPSI = mkrs['RPSI']
    LPSI = mkrs['LPSI']

    # Create versors
    O = (RASI + LASI) / 2
    Opel = O.copy()
    Zpel = getVersor(RASI - LASI)
    if s == 'R':
        Ypel = getVersor(np.cross(Zpel, RASI - 0.5 * (LPSI + RPSI)))
    else:
        Ypel = getVersor(np.cross(Zpel, LASI - 0.5 * (LPSI + RPSI)))
    Xpel = getVersor(np.cross(Ypel, Zpel))

    # Create rotation matrix from shank reference frame to laboratory reference frame
    R = np.array((Xpel.T, Ypel.T, Zpel.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Opel
    
    
def pelvisPoseNoOneASI(mkrs, s='R'):
    """Calculate roto-translation matrix from pelvis (ISB conventions) to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'RASI': right spina iliaca anterior superior
        - 'LASI': right spina iliaca anterior superior
        - 'RPSI': right spina iliaca posterior superior
        - 'LPSI': right spina iliaca posterior superior

    s : {'R', 'L'}
        Anatomical side.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Wu G, Siegler S, Allard P, Kirtley C, Leardini A, Rosenbaum D, Whittle M,
    D'Lima DD, Cristofolini L, Witte H, Schmid O, Stokes I; Standardization and
    Terminology Committee of the International Society of Biomechanics. ISB
    recommendation on definitions of joint coordinate system of various joints for
    the reporting of human joint motion--part I: ankle, hip, and spine. International
    Society of Biomechanics. J Biomech. 2002 Apr;35(4):543-8. PubMed PMID: 11934426.

    """

    # Define markers to use
    RASI = mkrs['RASI'] if 'RASI' in mkrs else None
    LASI = mkrs['LASI'] if 'LASI' in mkrs else None
    RPSI = mkrs['RPSI']
    LPSI = mkrs['LPSI']

    # Create versors
    O = (RPSI + LPSI) / 2
    Opel = O.copy()
    Zpel = getVersor(RPSI - LPSI)
    if s == 'R':
        Ypel = getVersor(np.cross(Zpel, (RASI if RASI is not None else LASI) - 0.5 * (LPSI + RPSI)))
    else:
        Ypel = getVersor(np.cross(Zpel, (RASI if RASI is not None else LASI) - 0.5 * (LPSI + RPSI)))
    Xpel = getVersor(np.cross(Ypel, Zpel))

    # Create rotation matrix from shank reference frame to laboratory reference frame
    R = np.array((Xpel.T, Ypel.T, Zpel.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Opel


def shankPoseISB(mkrs, s='R'):
    """Calculate roto-translation matrix from shank (ISB conventions) to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'MM': medial malleolus
        - 'LM': lateral melleolus
        - 'HF': head of fibula
        - 'TT': tibial tuberosity

    s : {'R', 'L'}
        Anatomical side.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Leardini A, Benedetti MG, Berti L, Bettinelli D, Nativo R, Giannini S.
    Rear-foot, mid-foot and fore-foot motion during the stance phase of gait.
    Gait Posture. 2007 Mar;25(3):453-62. Epub 2006 Sep 11. PubMed PMID: 16965916.

    """

    # Define markers to use
    MM = mkrs['MM']
    LM = mkrs['LM']
    HF = mkrs['HF']
    TT = mkrs['TT']

    # Create versors
    IM = (LM + MM) / 2
    Osha = IM.copy()
    if s == 'R':
        XshaTemp = getVersor(np.cross(Osha - LM, HF - LM))
    else:
        XshaTemp = -getVersor(np.cross(Osha - LM, HF - LM))
#    Ysha = getVersor((TT - Osha) - np.multiply(XshaTemp,vdot2(TT - Osha, XshaTemp)))
    Ysha = getVersor((TT - Osha) - XshaTemp * vdot2(TT - Osha, XshaTemp)[:,None])
    Zsha = getVersor(np.cross(XshaTemp, Ysha))
    Xsha = getVersor(np.cross(Ysha, Zsha))

    # Create rotation matrix from shank reference frame to laboratory reference frame
    R = np.array((Xsha.T, Ysha.T, Zsha.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Osha


def shankPoseISBWithClusterSVD(mkrs, clusterMkrList, args):
    """Calculate roto-translation matrix from shank (ISB conventions) to
    laboratory reference frame, using rigid segment-connected cluster of
    technical markers.

    Parameters
    ----------
    mkrs : dict
        Technical markers data. Keys are marker names, values are np.ndarray
        N x 3, where N is the number of time frames.

    clusterMkrList : list
        List of technical marker names to use.

    args : mixed
        Additional arguments:

        - 'mkrsLoc': dictionary where keys are marker names and values are
          3-elem np.arrays indicating the coordinates in the local reference
          frame. Both technical and anatomical markers are needed here. For
          For anatomical landmark names, see ``shankPoseISB()``.
        - 'side': anatomical side, 'R' or 'L'.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    mkrsSeg : dict
        Anatomical markers data in the laboratory reference frame.

    """

    # Get roto-translation matrix from cluster to laboratory reference frame
    R, T, info = rigidBodySVDFun(mkrs, clusterMkrList, args)
    gRl = composeRotoTranslMatrix(R, T)

    # Get markers in local reference frame
    mkrsLoc = {m: args['mkrsLoc'][m] for m in ['LM','MM','HF','TT']}

    # Express markers in the global reference frame
    mkrsSeg = changeMarkersReferenceFrame(mkrsLoc, gRl)

    # Calculate roto-translation matrix from shank to laboratory reference frame
    R, T = shankPoseISB(mkrsSeg, s=args['side'])

    return R, T, mkrsSeg


def footPoseISB(mkrs, s='R'):
    """Calculate roto-translation matrix from foot (ISB conventions) to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'CA': calcalneous
        - 'FM': first metatarsal head
        - 'SM': second metatarsal head
        - 'VM': fifth metatarsal head

    s : {'R', 'L'}
        Anatomical side.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Leardini A, Benedetti MG, Berti L, Bettinelli D, Nativo R, Giannini S.
    Rear-foot, mid-foot and fore-foot motion during the stance phase of gait.
    Gait Posture. 2007 Mar;25(3):453-62. Epub 2006 Sep 11. PubMed PMID: 16965916.

    """

    # Define markers to use
    CA = mkrs['CA']
    FM = mkrs['FM']
    SM = mkrs['SM']
    VM = mkrs['VM']

    # Create versors
    Ofoo = CA.copy()
    if s == 'R':
        YfooTemp = getVersor(np.cross(VM - Ofoo, FM - Ofoo))
    else:
        YfooTemp = -getVersor(np.cross(VM - Ofoo, FM - Ofoo))
#    Xfoo = getVersor((SM - Ofoo) - np.multiply(YfooTemp,vdot2(SM - Ofoo, YfooTemp)))
    Xfoo = getVersor((SM - Ofoo) - YfooTemp * vdot2(SM - Ofoo, YfooTemp)[:,None])
    Zfoo = getVersor(np.cross(Xfoo, YfooTemp))
    Yfoo = getVersor(np.cross(Zfoo, Xfoo))

    # Create rotation matrix from foot reference frame to laboratory reference frame
    R = np.array((Xfoo.T, Yfoo.T, Zfoo.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Ofoo


def footPoseISBWithClusterSVD(mkrs, clusterMkrList, args):
    """Calculate roto-translation matrix from foot (ISB conventions) to
    laboratory reference frame, using rigid segment-connected cluster of
    technical markers.

    Parameters
    ----------
    mkrs : dict
        Technical markers data. Keys are marker names, values are np.ndarray
        N x 3, where N is the number of time frames.

    clusterMkrList : list
        List of technical marker names to use.

    args : mixed
        Additional arguments:

        - 'mkrsLoc': dictionary where keys are marker names and values are
          3-elem np.arrays indicating the coordinates in the local reference
          frame. Both technical and anatomical markers are needed here. For
          For anatomical landmark names, see ``footPoseISB()``.
        - 'side': anatomical side, 'R' or 'L'.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    mkrsSeg : dict
        Anatomical markers data in the laboratory reference frame.

    """

    # Get roto-translation matrix from cluster to laboratory reference frame
    R, T, info = rigidBodySVDFun(mkrs, clusterMkrList, args)
    gRl = composeRotoTranslMatrix(R, T)

    # Get markers in local reference frame
    mkrsLoc = {m: args['mkrsLoc'][m] for m in ['CA','FM','SM','VM']}

    # Express markers in the global reference frame
    mkrsSeg = changeMarkersReferenceFrame(mkrsLoc, gRl)

    # Calculate roto-translation matrix from foot to laboratory reference frame
    R, T = footPoseISB(mkrsSeg, s=args['side'])

    return R, T, mkrsSeg


def calcaneusPose(mkrs, s='R'):
    """Calculate roto-translation matrix from calcaneous to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'CA': calcalneous
        - 'PT': lateral apex of the peroneal tubercle
        - 'ST': most medial apex of the sustentaculum tali

    s : {'R', 'L'}
        Anatomical side.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Leardini A, Benedetti MG, Berti L, Bettinelli D, Nativo R, Giannini S.
    Rear-foot, mid-foot and fore-foot motion during the stance phase of gait.
    Gait Posture. 2007 Mar;25(3):453-62. Epub 2006 Sep 11. PubMed PMID: 16965916.

    """

    # Define markers to use
    CA = mkrs['CA']
    PT = mkrs['PT']
    ST = mkrs['ST']

    # Create versors
    IC = (ST + PT) / 2
    Ocal = CA.copy()
    Xcal = getVersor(IC - Ocal)
    if s == 'R':
        YcalTemp = getVersor(np.cross(Xcal, ST - Ocal))
    else:
        YcalTemp = -getVersor(np.cross(Xcal, ST - Ocal))
    Zcal = getVersor(np.cross(Xcal, YcalTemp))
    Ycal = getVersor(np.cross(Zcal, Xcal))

    # Create rotation matrix from foot reference frame to laboratory reference frame
    R = np.array((Xcal.T, Ycal.T, Zcal.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Ocal


def calcaneusPoseWithClusterSVD(mkrs, clusterMkrList, args):
    """Calculate roto-translation matrix from calcaneous to
    laboratory reference frame, using rigid segment-connected cluster of
    technical markers.

    Parameters
    ----------
    mkrs : dict
        Technical markers data. Keys are marker names, values are np.ndarray
        N x 3, where N is the number of time frames.

    clusterMkrList : list
        List of technical marker names to use.

    args : mixed
        Additional arguments:

        - 'mkrsLoc': dictionary where keys are marker names and values are
          3-elem np.arrays indicating the coordinates in the local reference
          frame. Both technical and anatomical markers are needed here. For
          For anatomical landmark names, see ``calcaneusPoseISB()``.
        - 'side': anatomical side, 'R' or 'L'.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    mkrsSeg : dict
        Anatomical markers data in the laboratory reference frame.

    """

    # Get roto-translation matrix from cluster to laboratory reference frame
    R, T, info = rigidBodySVDFun(mkrs, clusterMkrList, args)
    gRl = composeRotoTranslMatrix(R, T)

    # Get markers in local reference frame
    mkrsLoc = {m: args['mkrsLoc'][m] for m in ['CA','PT','ST']}

    # Express markers in the global reference frame
    mkrsSeg = changeMarkersReferenceFrame(mkrsLoc, gRl)

    # Calculate roto-translation matrix from foot to laboratory reference frame
    R, T = calcaneusPose(mkrsSeg, s=args['side'])

    return R, T, mkrsSeg
    
    
def humerusPose(mkrs, s='R'):
    """Calculate roto-translation matrix from humerus to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'GH': glenohumeral rotation center
        - 'LE': most caudal point on lateral epicondyle
        - 'ME': most caudal point on medial epicondyle

    s : {'R', 'L'}
        Anatomical side.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Wu G, van der Helm FC, Veeger HE, Makhsous M, Van Roy P, Anglin C, Nagels J,
    Karduna AR, McQuade K, Wang X, Werner FW, Buchholz B; International Society of
    Biomechanics. ISB recommendation on definitions of joint coordinate systems of
    various joints for the reporting of human joint motion--Part II: shoulder, elbow,
    wrist and hand. J Biomech. 2005 May;38(5):981-992. Review. PubMed PMID: 15844264.

    """

    # Define markers to use
    GH = mkrs['GH']
    LE = mkrs['LE']
    ME = mkrs['ME']

    # Create versors
    E = (LE + ME) / 2
    Ohum = GH.copy()
    if s == 'R':
        Yhum = getVersor(GH - E)
    else:
        Yhum = -getVersor(GH - E)
    if s == 'R':
        Xhum = getVersor(np.cross(Yhum, LE - ME))
    else:
        Xhum = -getVersor(np.cross(Yhum, LE - ME))
    Zhum = getVersor(np.cross(Xhum, Yhum))

    # Create rotation matrix from foot reference frame to laboratory reference frame
    R = np.array((Xhum.T, Yhum.T, Zhum.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Ohum
    
    
def scapulaPose(mkrs, s='R'):
    """Calculate roto-translation matrix from scapula to
    laboratory reference frame.

    Parameters
    ----------
    mkrs : dict
        Markers data. Keys are marker names, values are np.ndarray N x 3,
        where N is the number of time frames. Used names are:

        - 'AA': angulus acromialis (acromial angle)
        - 'AI': angulus inferior (inferior angle),
        - 'TS': trigonum spinae scapulae (root of the spine)

    s : {'R', 'L'}
        Anatomical side.

    Returns
    -------
    R : np.ndarray
        N x 3 x 3 rotation matrix.

    T : np.ndarray
        N x 3 translation vector.

    References
    ----------
    Wu G, van der Helm FC, Veeger HE, Makhsous M, Van Roy P, Anglin C, Nagels J,
    Karduna AR, McQuade K, Wang X, Werner FW, Buchholz B; International Society of
    Biomechanics. ISB recommendation on definitions of joint coordinate systems of
    various joints for the reporting of human joint motion--Part II: shoulder, elbow,
    wrist and hand. J Biomech. 2005 May;38(5):981-992. Review. PubMed PMID: 15844264.

    """

    # Define markers to use
    AA = mkrs['AA']
    IA = mkrs['IA']
    TS = mkrs['TS']

    # Create versors
    Osca = AA.copy()
    if s == 'R':
        Zsca = getVersor(AA - TS)
    else:
        Zsca = -getVersor(AA - TS)
    if s == 'R':
        Xsca = -getVersor(np.cross(Zsca, AA - IA))
    else:
        Xsca = getVersor(np.cross(Zsca, AA - IA))
    Ysca = getVersor(np.cross(Zsca, Xsca))

    # Create rotation matrix from foot reference frame to laboratory reference frame
    R = np.array((Xsca.T, Ysca.T, Zsca.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    # Return data
    return R, Osca



def ges(Rvect):
    """Calculate Groot & Suntay anatomical joint angles from proximal and distal
    segment rotation matrices. Angles are related to flexion-extension (FE) axis
    of the proximal segment, internal-external (IE) axis of the distal segment,
    ab-adduction (AA) floating axis.

    Parameters
    ----------
    Rvect : np.ndarray
        18-elem vector representing row-flattened version of proximal and
        distal segment rotation matrix from global reference frame to segment.

    Returns
    -------
    list
        List of Groot & Suntay angles (FE, AA, EI).

    References
    ----------
    Grood et Suntay, A joint coordinate system for the clinical description of
    three- dimensional motion: application to the knee.
    J Biomech. Engng 1983 105: 136-144
    """

    R1v = Rvect[0:9]
    R2v = Rvect[9:18]
    e2=np.cross(R2v[3:6],R1v[6:9]) # e2 = e3 x e1
    #---- i/e rotation ----
    e2zd=np.dot(e2,-R2v[6:9])
    e2xd=np.dot(e2,R2v[0:3])
    IE=-np.arctan2(e2zd,e2xd)
    #---- flexion  ----
    e2yp=np.dot(e2,R1v[3:6])
    e2xp=np.dot(e2,R1v[0:3])
    FE=np.arctan2(e2yp,e2xp)
    #---- ab-adduction ----
    bet=np.dot(R2v[3:6],R1v[6:9])
    AA=np.arccos(bet)-np.pi/2
    res = FE, -AA, IE
    return res


def R2zxy(Rvect):
    """Convert joint rotation matrix to ZXY Euler sequence.

    Parameters
    ----------
    Rvect : np.ndarray
        A 9-elements array representing concatenated rows of the joint
        rotation matrix.

    Returns
    -------
    list
        A list of 3 angle values.

    """

    row1 = Rvect[0:3]
    row2 = Rvect[3:6]
    row3 = Rvect[6:9]
    R = np.matrix([row1,row2,row3]) # 3 x 3 joint rotation matrix
    x1 = np.arcsin(R[2,1])
    sy =-R[2,0]/np.cos(x1)
    cy = R[2,2]/np.cos(x1)
    y1 = np.arctan2(sy,cy)
    sz =-R[0,1]/np.cos(x1)
    cz = R[1,1]/np.cos(x1)
    z1 = np.arctan2(sz,cz)
    if x1 >= 0:
      x2 = np.pi - x1
    else:
      x2 = -np.pi - x1
    sy =-R[2,0]/np.cos(x2)
    cy = R[2,2]/np.cos(x2)
    y2 = np.arctan2(sy,cy)
    sz =-R[0,1]/np.cos(x2)
    cz = R[1,1]/np.cos(x2)
    z2 = np.arctan2(sz,cz)
    if -np.pi/2 <= x1 and x1 <= np.pi/2:
      yAngle=y1
      zAngle=z1
      xAngle=x1
    else:
      yAngle=y2
      zAngle=z2
      xAngle=x2
    res = zAngle, xAngle, yAngle
    return res
    
    
def R2yxyKS(Rvect):
    """Convert joint rotation matrix to YXY Euler sequence.

    Parameters
    ----------
    Rvect : np.ndarray
        A 9-elements array representing concatenated rows of the joint
        rotation matrix.

    Returns
    -------
    list
        A list of 3 angle values.

    """

    row1 = Rvect[0:3]
    row2 = Rvect[3:6]
    row3 = Rvect[6:9]
    R = np.matrix([row1,row2,row3]) # 3 x 3 joint rotation matrix
    ya, x, y = rot.R_to_euler_yxy(np.linalg.inv(R))
    
    return y, x, ya
    
    
def R2yxy(Rvect):
    """Convert joint rotation matrix to YXY Euler sequence.

    Parameters
    ----------
    Rvect : np.ndarray
        A 9-elements array representing concatenated rows of the joint
        rotation matrix.

    Returns
    -------
    list
        A list of 3 angle values.

    """

    row1 = Rvect[0:3]
    row2 = Rvect[3:6]
    row3 = Rvect[6:9]
    R = np.matrix([row1,row2,row3]) # 3 x 3 joint rotation matrix
    R = np.linalg.inv(R)

    x = np.arccos(R[1,1])
    y = np.arctan2(R[0,1], R[2,1])
    ya = np.arctan2(R[1,0], -R[1,2])
    
    return y, x, ya


def getJointAngles(R1, R2, R2anglesFun=R2zxy, funInput='jointR', **kwargs):
    """Calculate 3 joint angles between 2 rigid bodies.

    Parameters
    ----------
    R1 : np.ndarray
        N x 3 x 3 rotation matrices from rigid body to global reference frame for body 1 (N time frames).

    R2 : np.ndarray
        N x 3 x 3 rotation matrices from rigid body to global reference frame for body 2.

    R2anglesFun : func
        Function converting from joint rotation matrix, or segment matrices, to angles (see ``R2zxy()``, ``ges()``).

    funInput : str
        Input type for R2anglesFun.
        If 'jointR', the input for the function is a a 9-elements array representing
        concatenated rows of the joint rotation matrix.
        If 'segmentsR', the input for the function is a a 18-elements array representing
        concatenated rows of the rotation matrices for the 2 segments.

    **kwargs
        Any further argument to R2anglesFun.

    Returns
    -------
    np.ndarray
        N x 3 matrix of angles (in *deg*)

    """

    N, dim1, dim2 = R1.shape
    if funInput == 'jointR':
        Rj = dot3(inv2(R1), R2)
        Rjv = np.squeeze(np.reshape(Rj,(N,9)))
        if len(Rjv.shape) == 1:
            Rjv = Rjv[None,:]
        angles = np.apply_along_axis(R2anglesFun, 1, Rjv, **kwargs)
    else:
        R1v = np.squeeze(np.reshape(inv2(R1),(N,9)))
        R2v = np.squeeze(np.reshape(inv2(R2),(N,9)))
        Rvect = np.hstack((R1v,R2v))
        if len(Rvect.shape) == 1:
            Rvect = Rvect[None,:]
        angles = np.apply_along_axis(R2anglesFun, 1, Rvect, **kwargs)
    # Correct for gimbal-lock
    angles[:,0] = correctGimbal(angles[:,0])
    angles[:,1] = correctGimbal(angles[:,1])
    angles[:,2] = correctGimbal(angles[:,2])
    return np.rad2deg(angles)
    
    
def correctGimbal(angle):
    angle2 = angle[:]
    th = 1.5 * np.pi
    for i in range(1, angle2.shape[0]):
        if (angle2[i] - angle2[i-1]) >= th:
            angle2[i:] -= 2 * np.pi
        elif (angle2[i] - angle2[i-1]) <= -th:
            angle2[i:] += 2 * np.pi
    return angle2


def getJointTransl(R1, R2, O1, O2, T2translFun=None, **kwargs):
    """Calculate 3 translations between 2 rigid bodies.

    Parameters
    ----------
    R1 : np.ndarray
        N x 3 x 3 rotation matrices from rigid body to global reference frame for body 1 (N time frames).

    R2 : np.ndarray
        N x 3 x 3 rotation matrices from rigid body to global reference frame for body 2.

    T2translFun : func
        Function converting from segment poses to angles.

    O1 : np.ndarray
        N x 3 translation vectors from rigid body to global reference frame for body 1.

    O2 : np.ndarray
        N x 3 translation vectors from rigid body to global reference frame for body 2.

    **kwargs
        Any further argument to T2translFun.

    Returns
    -------
    np.ndarray
        N x 3 matrix of translations.

    """

    N, dim1, dim2 = R1.shape
    R1v = np.squeeze(np.reshape(inv2(R1),(N,9)))
    R2v = np.squeeze(np.reshape(inv2(R2),(N,9)))
    O1v = np.squeeze(np.reshape(O1,(N,3)))
    O2v = np.squeeze(np.reshape(O2,(N,3)))
    Tvect = np.hstack((R1v,R2v,O1v,O2v))
    if len(Tvect.shape) == 1:
        Tvect = Tvect[None,:]
    transl = np.apply_along_axis(T2translFun, 1, Tvect, **kwargs)
    return transl
    
    
def gesTranslFromSegment1(Tvect, **kwargs):

    Rvect = Tvect[0:18]
    O1, O2 = Tvect[18:21], Tvect[21:24]

    R1v = Rvect[0:9]

    x1, y1, z1 = R1v[0:3], R1v[3:6], R1v[6:9]

    v = O1 - O2
    Tx = np.dot(v, x1)
    Ty = np.dot(v, y1)
    Tz = np.dot(v, z1)

    return Tx, Ty, Tz
    
    
def lineFitSVD(points):
    # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    # Calculate center and versor
    centroid = points.mean(axis=0)
    uu, dd, vv = np.linalg.svd(points - centroid)
    versor = vv[0]
    center = centroid
    # Calculate other data
    proj = np.dot(points - centroid, versor)
    other = {}
    other['proj'] = proj
#    print (points - centroid)[:10,:]
#    print np.linalg.norm(points - centroid, axis=1)[:10]
#    print proj[:10]
    return versor, center, other
    
    
def planeFitSVD(points):
    # http://stackoverflow.com/questions/15959411/fit-points-to-a-plane-algorithms-how-to-iterpret-results
    # Calculate center and normal
    centroid = points.mean(axis=0)
    uu, dd, vv = np.linalg.svd(points - centroid)
    versor = vv[2]
    center = centroid
    # Calculate other data
    proj = points - (np.dot(points - centroid, versor))[:,np.newaxis] * versor
    other = {}
    other['proj'] = proj
    return versor, center, other
