"""
.. module:: vtkh
   :synopsis: helper module for VTK

"""

import vtk


def reposeVTKData(vktDataIn, pose):
    transform = vtk.vtkTransform()
    transform.SetMatrix(pose.ravel().tolist())
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInput(vktDataIn)
    transformFilter.Update()
    vtkDataOut = transformFilter.GetOutput()
    return vtkDataOut
    
    
    
def createVTKActor(vtkData, color=None):
    mapper = vtk.vtkPolyDataMapper()
    if vtkData.GetClassName() == 'vtkPolyData':
        vtkPolyData = vtkData
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(vtkPolyData)
        else:
            mapper.SetInputData(vtkPolyData)
            mapper.Update()
    elif vtkData.GetClassName() == 'vtkSphereSource':
        source = vtkData
        mapper.SetInput(source.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if color:
        actor.GetProperty().SetColor(*color)
    return actor
    
    
    
def showVTKActors(actors):
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
     
    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
    

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
    return source
        