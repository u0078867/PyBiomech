"""
.. module:: kine_wr
   :synopsis: helper module for kinematics for wrist-rig

"""


import numpy as np
from .kine import getVersor

    


def handWR(mkrs, s='R'):

    H3MC = mkrs['H3MC']
    B3MC = mkrs['B3MC']
    H2MC = mkrs['H2MC']

    O = H3MC.copy()
    
    Y = getVersor(B3MC - H3MC)
    if s == 'L':
        Y = -Y
    X = getVersor(np.cross(H2MC - B3MC, H3MC - B3MC))
    Z = getVersor(np.cross(X, Y))
    
    I = Z
    K = Y
    J = X
    
    R = np.array((J.T, K.T, I.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    return R, O



def forearmWR(mkrs, s='R'):

    US = mkrs['US']
    midE = mkrs['midE']
    RS = mkrs['RS']

    O = US.copy()
    
    Y = getVersor(midE - US)
    if s == 'L':
        Y = -Y
    X = getVersor(np.cross(RS - midE, US - midE))
    Z = getVersor(np.cross(X, Y))
    
    I = Z
    K = Y
    J = X
    
    R = np.array((J.T, K.T, I.T))   # 3 x 3 x N
    R = np.transpose(R, (2,1,0))  # N x 3 x 3

    return R, O





