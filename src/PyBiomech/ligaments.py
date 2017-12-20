# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 10:35:47 2017

@author: u0078867
"""

import vtk
from vtkh import *
import numpy as np
from scipy.optimize import minimize as spminimize
from scipy.interpolate import RegularGridInterpolator

  
    
def calcLigaLength(ligaPath) :
    L = np.linalg.norm(np.diff(ligaPath, axis=0), axis=1).sum()
    return L
    
    
def ligamentPathBlankevoort1991(pIns, edgeSpline):
    # Evaluate spline in 100 points
    p1 = pIns[0,:]
    p2 = pIns[1,:]
    unew = np.arange(0, 1.01, 0.01)
    tck = edgeSpline
    out = interpolate.splev(unew, tck)
    pe = np.array(out).T
    
    # Find edge point minimizing total length
    L = np.linalg.norm(p1-pe, axis=1) + np.linalg.norm(pe-p2, axis=1)
    iMin = np.argmin(L)
    pem = pe[iMin,:]
    pl = np.array((p1,pem,p2))

    # Check if straight line is not penetrating
    pe1 = pe[0,:]
    pe2 = pe[-1,:]
    pup = np.cross(pe1 - pem, pe2 - pem)
    if np.dot(p1 - pem, pup) < 0:
        pe1, pe2 = pe2, pe1
    if np.dot(np.cross(p1 - pem, p2 - pem), pe2 - pe1) < 0:
        #print('straight line not penetrating')
        pl = np.array((p1,p2))
        
    return pe, pl
    
    
    
def ligamentPathMarai2004(pIns, vtkBoneA, vtkBoneB, Ns=10, iterInitP=[], iterArgs={}, equalizeInitIterP=True):
    # Unpack insertion points
    p1 = pIns[0,:]
    p2 = pIns[1,:]
    
    # Create pose matrin from global to straight ligament reference frame
    z = p1 - p2
    z = z / np.linalg.norm(z)
    x = np.cross(z, (1, 0, 0))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    R = np.array((x, y, z)).T
    Rt = np.vstack((np.hstack((R, p2[:,None])), (0, 0, 0, 1)))
    
    # Repose all elements in new reference frame
    vtkBoneA2 = reposeVTKData(vtkBoneA, Rt.ravel())
    vtkBoneB2 = reposeVTKData(vtkBoneB, Rt.ravel())
    p12 = np.dot(Rt, np.append(p1, 1))[:3]
    p22 = np.dot(Rt, np.append(p2, 1))[:3]
    if Ns is None:
        iterInitP2 = np.dot(Rt, np.vstack((iterInitP.T, np.ones((1, iterInitP.shape[0]))))).T[:,:3]
        print p12
        print p22
        print iterInitP2
    
    # Create points
    if Ns is not None:
        L = np.arange(Ns + 1)
        V = (p12 - p22) / Ns
        p = np.zeros((Ns + 1, 3))
        for l in L:
            ps2 = p22 + l * V
            p[l,:] = ps2
    else:
        p = iterInitP2.copy()[::-1,:]
        Ns = p.shape[0] - 1
        if equalizeInitIterP:
            print p
#            p = p[::-1,:]
            xuf = RegularGridInterpolator((p[:,2],), p[:,0], bounds_error=False, fill_value=None)
            yuf = RegularGridInterpolator((p[:,2],), p[:,1], bounds_error=False, fill_value=None)
            Ns = p.shape[0] - 1
            L = np.arange(Ns + 1)
            Z = (p12[2] - p22[2]) / Ns
            for l in L:
                z = p22[2] + l * Z
                x = xuf([[z]])[0]
                y = yuf([[z]])[0]
                ps2 = (x, y, z)
                p[l,:] = ps2
#            p = p[::-1,:]
            print p
    
    points = vtk.vtkPoints()
    for ps2 in p:
        points.InsertNextPoint(ps2)
    checkPoints = vtk.vtkPolyData()
    checkPoints.SetPoints(points)
    
    # Create points checker
    penetration = arePointsPenetrating(vtkBoneA2, checkPoints) or \
                    arePointsPenetrating(vtkBoneB2, checkPoints)
    print('Penetration: %d' % penetration)
                    
    if penetration:
    
        # Create points distancer
        pointDistancer = vtk.vtkImplicitPolyDataDistance()
        
        # Minimize length
        def retablePoints2C(x):
            # reorganize via points
            xv = x.reshape((Ns-1,2))
            # reconstruct (Ns+1) x 3 all points array
            pp = np.vstack((p22[0:2],xv,p12[0:2]))
            pp = np.hstack((pp, p[:,2:3]))
            return pp
        
        retablePoints = retablePoints2C
        
        def fun(x):
            pp = retablePoints(x)
            d = np.linalg.norm(pp[1:,:] - pp[:-1,:], axis=1)
            dtot = d.sum()
            print('Current length: %f' % dtot)
            return dtot
    
        x0 = p[1:-1,0:2].ravel()
        
        print('-- Initial condition:')
        fun(x0)
        
        def consFun(x, ip, bone):
            pp = retablePoints(x)
            if bone == 'boneB':
                vtkBone = vtkBoneB2
            elif bone == 'boneA':
                vtkBone = vtkBoneA2
            pointDistancer.SetInput(vtkBone)
            d = pointDistancer.EvaluateFunction(pp[ip,:])
            return d
            
        cons = []
        for ip in xrange(1, Ns):
            cons.append({'type': 'ineq', 'fun': consFun, 'args': (ip, 'boneB')})
            cons.append({'type': 'ineq', 'fun': consFun, 'args': (ip, 'boneA')})
            
        def jac2C(x):
            pp = retablePoints(x)
            Nv = pp.shape[0] - 2
            jaco = np.zeros((Nv, 2))
            dx = pp[1:,0:2] - pp[:-1,0:2]
            dist = np.linalg.norm(dx, axis=1)
            for i in xrange(0, Nv):
                jaco[i,:] = (dx[i,:] / dist[i]) - (dx[i+1,:] / dist[i+1])
            return jaco.ravel()
            
        def jac2Cc(x):
            pp = retablePoints(x)
            Nv = pp.shape[0] - 2
            jaco = np.zeros((Nv, 3))
            dx = pp[1:,:] - pp[:-1,:]
            dist = np.linalg.norm(dx, axis=1)
            for i in xrange(0, Nv):
                jaco[i,:] = (dx[i,:] / dist[i]) - (dx[i+1,:] / dist[i+1])
            return jaco[:,:2].ravel()
        
        jac = jac2Cc
        
        if not iterArgs: # if it is empty
            options = {'disp': True, 'eps' : 50e0, 'maxiter': 20}
            #options = {'disp': True, 'eps' : 50e0, 'maxiter': 0}
        else:
            options = iterArgs
        print('-- Iterative algorithm:')
        res = spminimize(
            fun, 
            x0, 
            method='SLSQP', 
            constraints=cons,
            jac=jac,
            options=options
        )
        print('-- Final condition:')
        fun(res.x)
        pw = retablePoints(res.x)
        pw = pw[::-1,:]
        p = p[::-1,:]
        print(pw - p)
        print(pw)
        
        # Repose all points in original global reference frames
        p = np.dot(np.linalg.inv(Rt), np.vstack((p.T, np.ones((1, p.shape[0]))))).T[:,:3]
        pw = np.dot(np.linalg.inv(Rt), np.vstack((pw.T, np.ones((1, pw.shape[0]))))).T[:,:3]
        
    else:
        
        p = pIns.copy()
        pw = p.copy()
    
    return p, pw
    
        
    

def estimateKneeLigamentPath(
                        pIns,
                        method='blankevoort_1991',
                        vtkFemur=None,
                        vtkTibia=None,
                        methodArgs={},
                        show3DNavigator=True
    ):
        
    actors = []
        
    if method == 'marai_2004':
        
        # Unpack arguments
        iterArgs = methodArgs['iter_args']
        iterInitP = methodArgs['iter_init_p']
        showInitP = methodArgs['show_init_p']
        Ns = methodArgs['ns']
        if vtkFemur is None or vtkTibia is None:
            raise Exception('vtkFemur and vtkTibia needed for this method')
        # Run the algorithm
        pInit, pWrap = ligamentPathMarai2004(pIns, vtkFemur, vtkTibia, Ns, iterInitP, iterArgs=iterArgs)
        # Add initial ligament actor
        if show3DNavigator and showInitP:
            actors.append(createVTKActor(createLineVTKData(pInit, [255, 0, 0])))
        
    elif method == 'blankevoort_1991':
        
        edgeSpline = methodArgs['edge_spline']
        showEdgeSplineP = methodArgs['show_edge_spline_p']
        pEdgeSpline, pWrap = ligamentPathBlankevoort1991(pIns, edgeSpline)
        if show3DNavigator and showEdgeSplineP:
            actors.append(createVTKActor(createLineVTKData(pEdgeSpline, [255, 0, 0])))
    
    # Prepare 3D navigation environment
    if show3DNavigator:
        
        # Add femur actor
        if vtkFemur is not None:
            actors.append(createVTKActor(vtkFemur))
            
        # Add tibia actor
        if vtkFemur is not None:
            actors.append(createVTKActor(vtkTibia))
           
        # Add ligament actor
        actors.append(createVTKActor(createLineVTKData(pWrap, [0, 255, 0])))
    
    # Return ligament path
    return pWrap
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    