"""
.. module:: vtkh
   :synopsis: helper module for VTK

"""

import vtk
import fio
from vtk.util.numpy_support import vtk_to_numpy as v2n
import numpy as np
from scipy import interpolate


ew = vtk.vtkFileOutputWindow()
ew.SetFileName("vtk_errors.log")
ow = vtk.vtkOutputWindow()
ow.SendToStdErrOn()
ow.SetInstance(ew)

    
    
def arePointsPenetrating(vtkData, vtkPoints):
    pointChecker = vtk.vtkSelectEnclosedPoints()
    if vtk.VTK_MAJOR_VERSION <= 5:
        pointChecker.SetInput(vtkPoints)
    else:
        pointChecker.SetInputData(vtkPoints)
    if vtk.VTK_MAJOR_VERSION <= 5:
        pointChecker.SetSurface(vtkData)
    else:
        pointChecker.SetSurfaceData(vtkData)
    pointChecker.SetCheckSurface(1)
    pointChecker.SetTolerance(0)
    pointChecker.Update()
    insideArr = v2n(pointChecker.GetOutput().GetPointData().GetArray('SelectedPoints'))
    penetration = (insideArr.sum() > 0)
    return penetration
    


def reposeVTKData(vktDataIn, pose):
    transform = vtk.vtkTransform()
    transform.SetMatrix(pose.ravel().tolist())
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    if vtk.VTK_MAJOR_VERSION <= 5:
        transformFilter.SetInput(vktDataIn)
    else:
        transformFilter.SetInputData(vktDataIn)
    transformFilter.Update()
    vtkDataOut = transformFilter.GetOutput()
    return vtkDataOut
    
    
    
def createVTKActor(vtkPolyData, presets=None, color=None, lineWidth=None, scalarRange=None):
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(vtkPolyData)
    else:
        mapper.SetInputData(vtkPolyData)
    if presets == 'contour':
        mapper.SetScalarVisibility(True)
        mapper.SetScalarRange(scalarRange)
    if scalarRange is not None:
        mapper.SetScalarRange(scalarRange)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if color is not None:
        actor.GetProperty().SetColor(*color)
    if presets == 'contour':
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(lineWidth)
    return actor
    

def create2DScalarBarActor(data, title, nLabels=4):
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(data)
    scalarBar.SetTitle(title)
    scalarBar.SetNumberOfLabels(nLabels)
    return scalarBar
    
    
    
def showVTKActors(actors):
    showScene(createScene(actors))
    
    
def createScene(actors):
    
    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
     
    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
     
    # Assign actors to the renderer
    for actor in actors:
        ren.AddActor(actor)
    
    # Add elements to scene object
    scene = {}
    scene['renderer'] = ren
    scene['window'] = renWin
    scene['interactor'] = iren
    return scene
    

def showScene(scene):
    # Extract elements
    renWin = scene['window']
    iren = scene['interactor']
    
    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
    
def exportScene(scene, filePrefix, ext='x3d', names=[]):
    renWin = scene['window']
    ren = scene['renderer']
    if ext == 'x3d':
        writer = vtk.vtkX3DExporter()
        writer.SetInput(renWin)
        writer.SetFileName(filePrefix + '.x3d')
        writer.Update()
        writer.Write()
    elif ext == 'obj':
        writer = vtk.vtkOBJExporter()
        writer.SetFilePrefix(filePrefix)
        writer.SetInput(renWin)
        writer.Write()
    elif ext == 'vtm':
        actors = ren.GetActors()
        actors.InitTraversal()
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(actors.GetNumberOfItems())
        for i in xrange(actors.GetNumberOfItems()):
            actor = actors.GetNextItem()
            block = actor.GetMapper().GetInput()
            mb.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), names[i])
            mb.SetBlock(i, block)
        writer = vtk.vtkXMLMultiBlockDataWriter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(mb)
        else:
            writer.SetInputData(mb)
        writer.SetFileName(filePrefix + '.vtm')
        writer.Write()
    
    

def createLineVTKData(pts, col):
    points = vtk.vtkPoints()
    for p in pts:
        points.InsertNextPoint(p)
    lines = vtk.vtkCellArray()
    pl = vtk.vtkPolyLine()
    pl.GetPointIds().SetNumberOfIds(points.GetNumberOfPoints())
    for i in range(points.GetNumberOfPoints()):
        pl.GetPointIds().SetId(i, i)
    lines.InsertNextCell(pl)
    polydata = vtk.vtkPolyData()
    polydata.SetLines(lines)
    polydata.SetPoints(points)
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    colors.InsertNextTupleValue(col)
    polydata.GetCellData().SetScalars(colors)
    return polydata
    
    
    
def createSphereVTKData(center, radius):
    source = vtk.vtkSphereSource()
    source.SetCenter(*center)
    source.SetRadius(radius)
    source.Update()
    vtkSphere = source.GetOutput()
    return vtkSphere
    

def createContourVTKData(vtkData, nValues):
    scalarRange = vtkData.GetScalarRange()
    contours = vtk.vtkContourFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        contours.SetInput(vtkData)
    else:
        contours.SetInputData(vtkData)
    contours.GenerateValues(nValues, scalarRange)
    contours.Update()
    vtkContour = contours.GetOutput()
    return vtkContour
    

def createParamSpline(pts, k=3, retU=False):
    p = np.array(pts)
    pList = p.T.tolist()
    tck, u = interpolate.splprep(pList, s=0, k=k)
    spline = tck
    if retU:
        return spline, u
    else:
        return spline
        
        
def createPolynomial(pts, k=3, retU=False):
    x, y = pts[:,0], pts[:,1]
    try:
        z = np.polyfit(x, y, k)
        tck, u = interpolate.splprep([x.tolist(),x.tolist()], s=0, k=1)
        poly = z, x
    except:
        z = np.nan * np.ones((k+1,))
        poly = z, x
        u = np.nan * np.ones((x.shape[0],))
    if retU:
        return poly, u
    else:
        return poly
    


def reposeSpline(spline, pose):
    tck = spline
    Np = tck[1][0].shape[0]
    pc = np.array(tck[1] + [Np*[1]])
    pcr = np.dot(pose, pc)
    t2 = tck[0][:]
    c2 = (pcr[0,:], pcr[1,:], pcr[2,:])
    k2 = tck[2]
    tck2 = [t2, c2, k2]
    spline2 = tck2
    return spline2
    


def evalSpline(spline, u):
    tck = spline
    out = interpolate.splev(u, tck)
    p = np.array(out).T
    return p
    

def evalPolynomial(poly, u):
    z, x = poly
    f = np.poly1d(z)
    tck, dummy = interpolate.splprep([x.tolist(),x.tolist()], s=0, k=1)
    xU = np.array(interpolate.splev(u, tck)[1])
    out = f(xU)
    p = np.array([xU, out]).T
    return p
    
    
def evalSplineDerivative(spline, u, der=1):
    tck = spline
    out = interpolate.splev(u, tck, der=der)
    der = np.array(out).T
    return der
    
    
def evalPolynomialDerivative(poly, u, der=1):
    z, x = poly
    f = np.poly1d(z)
    f2 = np.polyder(f, m=der)
    tck, dummy = interpolate.splprep([x.tolist(),x.tolist()], s=0, k=1)
    xU = np.array(interpolate.splev(u, tck)[1])
    out = f2(xU)
    p = np.array([np.ones((x.shape[0],)), out]).T
    return p
    
    
    
def showData(data):
    # Create actors for each item
    actors = []
    for item in data:
        
        # Create actor
        if item['type'] == 'point':
            radius = 3
            if 'radius' in item:
                radius = item['radius']
            d = createSphereVTKData(item['coords'], radius)
            color = (1,0,0)
            if 'color' in item:
                color = item['color']
            actor = createVTKActor(d, color=color)
        if item['type'] == 'line':
            color = (255,0,0)
            if 'color' in item:
                color = item['color']
            d = createLineVTKData(item['coords'], color)
            actor = createVTKActor(d)
        if item['type'] == 'STL':
            d = fio.readSTL(item['filePath'])
            actor = createVTKActor(d)
        opacity = 1
        if 'opacity' in item:
            opacity = item['opacity']
        actor.GetProperty().SetOpacity(opacity)
            
        # Add actor
        actors.append(actor)
        
    # Show actors
    showVTKActors(actors)
        
        

def decimateVTKData(vtkData, reductionFactor):
    deci = vtk.vtkDecimatePro()
    if vtk.VTK_MAJOR_VERSION <= 5:
        deci.SetInput(vtkData)
    else:
        deci.SetInputData(vtkData)
    deci.SetTargetReduction(reductionFactor)
    deci.PreserveTopologyOn()
    deci.Update()
    decimated = deci.GetOutput()
    return decimated
        
        
def getBoundingBox(vtkData):
    boundingBoxFilter = vtk.vtkOutlineFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        boundingBoxFilter.SetInput(vtkData)
    else:
        boundingBoxFilter.SetInputData(vtkData)
    boundingBoxFilter.Update()
    boundingBox = boundingBoxFilter.GetOutput()
    return boundingBox
    

def scaleVTKDataAroundCenter(vtkData, scales):
    
    center = vtkData.GetCenter()
    
    transform1 = vtk.vtkTransform()
    transform1.Translate(-center[0],-center[1],-center[2])
    transform2 = vtk.vtkTransform()
    transform2.Scale(*scales)
    transform3 = vtk.vtkTransform()
    transform3.Translate(center[0],center[1],center[2])
    
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Identity()
    transform.Concatenate(transform1)
    transform.Concatenate(transform2)
    transform.Concatenate(transform3)
    
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    if vtk.VTK_MAJOR_VERSION <= 5:
        transformFilter.SetInput(vtkData)
    else:
        transformFilter.SetInputData(vtkData)
    transformFilter.Update()
    vtkDataScaled = transformFilter.GetOutput()
    
    return vtkDataScaled
    
    
def clipVTKDataWithBox(vtkData, bounds):
    planes = vtk.vtkPlanes()
    planes.SetBounds(bounds)
    clipper = vtk.vtkClipPolyData()
    if vtk.VTK_MAJOR_VERSION <= 5:
        clipper.SetInput(vtkData)
    else:
        clipper.SetInputData(vtkData)
    clipper.SetClipFunction(planes)
    clipper.SetValue(0.0)
    clipper.InsideOutOn()
    clipper.Update()
    vtkDataClipped = clipper.GetOutput()
    return vtkDataClipped
    

def VTKScalarData2Numpy(vtkPolyData):
    nodes = v2n(vtkPolyData.GetPoints().GetData())
    scalars = v2n(vtkPolyData.GetPointData().GetScalars())
    return nodes, scalars
    
    
    
    
    