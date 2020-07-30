"""
.. module:: kine_or
   :synopsis: helper module for kinematics for Oxford-Rig (IORT UZLeuven)

"""


import numpy as np
from .kine import getVersor, nonCollinear5PointsStylusFun


def gensegOR(mkrs):

    X = mkrs['X']
    Y = mkrs['Y']
    Z = mkrs['Z']
    C = mkrs['C']

    O = C.copy()

    I = getVersor(X - C)
    J = getVersor(Y - C)
    K = getVersor(Z - C)

    R = np.array((J.T, K.T, I.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    return R, O
    


def femurOR(mkrs, s='R'):

    FHC = mkrs['FHC']
    FKC = mkrs['FKC']
    FLCC = mkrs['FLCC']
    FMCC = mkrs['FMCC']

    O = FKC.copy()

    K = getVersor(FHC - FKC)
    if s == 'R':
        J = getVersor(np.cross(K, FLCC - FMCC))
    else:
        J = -getVersor(np.cross(K, FLCC - FMCC))
    I = getVersor(np.cross(J, K))

    R = np.array((J.T, K.T, I.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    return R, O



def tibiaORLW(mkrs, s='R'):

    TKC = mkrs['TKC']
    TAC = mkrs['TAC']
    TLCC = mkrs['TLCC']
    TMCC = mkrs['TMCC']

    O = TKC.copy()

    K = getVersor(TKC - TAC)
    if s == 'R':
        I = getVersor(np.cross(np.cross(K, TLCC - TMCC), K))
    else:
        I = -getVersor(np.cross(np.cross(K, TLCC - TMCC), K))
    J = getVersor(np.cross(K, I))

    R = np.array((J.T, K.T, I.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    return R, O



def tibiaOR(mkrs, s='R'):

    TKC = mkrs['TKC']
    TAC = mkrs['TAC']
    TLCC = mkrs['TLCC']
    TMCC = mkrs['TMCC']

    O = TKC.copy()

    K = getVersor(TKC - TAC)
    if s == 'R':
        J = getVersor(np.cross(K, TLCC - TMCC))
    else:
        J = -getVersor(np.cross(K, TLCC - TMCC))
    I = getVersor(np.cross(J, K))

    R = np.array((J.T, K.T, I.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    return R, O



def gesORLW(Rvect, **kwargs):

    # taken from original Labview code in Kine_calcul.vi

    s = kwargs['s']

    R1v, R2v = Rvect[0:9], Rvect[9:18]

    j1, k1, i1 = R1v[0:3], R1v[3:6], R1v[6:9]
    j2, k2, i2 = R2v[0:3], R2v[3:6], R2v[6:9]

    e1 = i1.copy()
    e2 = np.cross(k2, i1)
    e3 = k2.copy()

    alphas = np.arcsin(-np.dot(e2, k1).real)
    alphac = np.arccos(np.dot(e2, j1).real)
    if alphac > np.pi/2:
        flex = alphac
    else:
        flex = alphas
    ext = -flex

    beta = np.absolute(np.arccos(np.dot(k2, i1)))
    if s == 'R':
        beta = beta - np.pi/2
    else:
        beta = np.pi/2 - beta
    adduct = -beta

    sinGamma = np.dot(e2, i2)
    if s == 'R':
        sinGamma *= -1.
    gamma = np.pi/2 - np.absolute(np.arccos(sinGamma))
    extrarot = gamma
    intrarot = -extrarot

    return ext, adduct, intrarot



def gesOR(Rvect, **kwargs):

    # taken from original MATLAB code in TibiofemoralKinematics_modifiedGS.m

    s = kwargs['s']

    R1v, R2v = Rvect[0:9], Rvect[9:18]

    j1, k1, i1 = R1v[0:3], R1v[3:6], R1v[6:9]
    j2, k2, i2 = R2v[0:3], R2v[3:6], R2v[6:9]

    e1 = i1.copy()
    e2 = np.cross(k2, i1)
    e2 = e2 / np.linalg.norm(e2)
    e3 = k2.copy()

    alphas = np.arcsin(-np.dot(e2, k1))
    alphac = np.arccos(np.dot(e2, j1))
    if alphac > np.pi/2:
        flex = alphac
    else:
        flex = alphas
    ext = -flex

    beta = np.arccos(np.dot(k2, i1))
    if s == 'R':
        beta = beta - np.pi/2
    else:
        beta = np.pi/2 - beta
    adduct = -beta

    sinGamma = np.dot(e2, i2)
    if s == 'R':
        sinGamma *= -1.
    gamma = np.arcsin(sinGamma)
    extrarot = gamma
    intrarot = -extrarot

    return ext, adduct, intrarot



def gesTranslOR(Tvect, **kwargs):

    # taken from original Labview code in Kine_calcul.vi

    Rvect = Tvect[0:18]
    O1, O2 = Tvect[18:21], Tvect[21:24]

    R1v, R2v = Rvect[0:9], Rvect[9:18]

    j1, k1, i1 = R1v[0:3], R1v[3:6], R1v[6:9]
    j2, k2, i2 = R2v[0:3], R2v[3:6], R2v[6:9]

    e1 = i1.copy()
    e2 = np.cross(k2, i1)
    e2 = e2 / np.linalg.norm(e2)
    e3 = k2.copy()

    v = O1 - O2
    ML = np.dot(v, e1)
    AP = np.dot(v, e2)
    IS = np.dot(v, e3)

    return ML, AP, IS
    
    

def getPointerTipOR(mkrs, **kwargs):
    wand = {}
    wand['markers'] = ['WN', 'WW', 'WE', 'WM', 'WS']
    wand['pos'] = {
        'WN': np.array([-11.6192, -55.9967, -1730.3991]),
        'WW': np.array([36.3206, -50.4521, -1678.2392]),
        'WE': np.array([-53.2079, -54.3819, -1687.5598]),
        'WM': np.array([-8.2232, -52.401, -1656.313]),
        'WS': np.array([-5.4458, -51.5086, -1586.3156]),
        'Tip': np.array([-0.8278, -42.4799, -1451.0327]),
    }
    wand['dist'] = {
        'WN': 279.901310546,
        'WW': 230.361353611,
        'WE': 242.549791825,
        'WM': 205.652915696,
        'WS': 135.662472261,
    }
    wand['offPlaneDist'] = 4.7
    wand['algoSVD'] = 2
    wand['algoTrilat'] = 0
    tip = nonCollinear5PointsStylusFun(mkrs, wand, **kwargs)
    return tip
    


def get3StrainGagesSensorModel():
    sensor = {}
    sensor['pos'] = {}
    sensor['pos']['gage1_top'] = np.array([2.7051, 4.2051, 0.0])
    sensor['pos']['gage1_bottom'] = np.array([-4.2051, -2.7051, 0.0])
    sensor['pos']['gage2_top'] = np.array([-2.7051, 4.2051, 0.0])
    sensor['pos']['gage2_bottom'] = np.array([4.2051, -2.7051, 0.0])
    sensor['pos']['gage3_top'] = np.array([0.0, 5.0, 0.0])
    sensor['pos']['gage3_bottom'] = np.array([0.0, -5.0, 0.0])
    return sensor




