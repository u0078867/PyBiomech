"""
.. module:: spine
   :synopsis: helper module for spine

"""

from . import vtkh
import numpy as np



def create2DSpline(pts, u=np.arange(0, 1.01, 0.01), order=3):
    spline = vtkh.createParamSpline(pts, k=order)
    pts2 = vtkh.evalSpline(spline, u)
    return pts2
    
    
def calcSplineTangentSlopes(pts, u=np.arange(0, 1.01, 0.01), k=3):
    spline, uPts = vtkh.createParamSpline(pts, k=k, retU=True)
    if u == 'only_pts':
        u = uPts
    der = vtkh.evalSplineDerivative(spline, u, der=1)
    return der
    
    
def create2DPolynomial(pts, u=np.arange(0, 1.01, 0.01), order=3):
    poly = vtkh.createPolynomial(pts, k=order)
    pts2 = vtkh.evalPolynomial(poly, u)
    return pts2
    
    
def evalPolynomial(pts, u, k=3):
    poly = vtkh.createPolynomial(pts, k=k, retU=False)
    val = vtkh.evalPolynomial(poly, u)
    return val
    
    
def calcPolynomialTangentSlopes(pts, u=np.arange(0, 1.01, 0.01), k=3):
    poly, uPts = vtkh.createPolynomial(pts, k=k, retU=True)
    if u == 'only_pts':
        u = uPts
    der = vtkh.evalPolynomialDerivative(poly, u, der=1)
    return der
    
    
def calcPolynomialDerivatives(pts, u=np.arange(0, 1.01, 0.01), k=3, der=1):
    poly, uPts = vtkh.createPolynomial(pts, k=k, retU=True)
    if u == 'only_pts':
        u = uPts
    der = vtkh.evalPolynomialDerivative(poly, u, der=der)
    return der
    
    
def calcInterlinesAngle(m1, m2):
    angles = np.rad2deg(np.arctan(np.abs((m1 - m2) / (1 + m1 * m2))))
    return angles
    
    
    
    