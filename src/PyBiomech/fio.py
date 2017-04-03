"""
.. module:: fio
   :synopsis: file read/write module

"""

import btk
import vtk
import re
import numpy as np


def readC3D(fileName, sections, opts={}):
    """Read C3D file.

    Parameters
    ----------
    fileName : str
        Full path of the C3D file.

    sections : list
        List of strings indicating which section to read.
        It can contain the following: 'markers'.

    opts : dict
        Options dictionary that can contain the following keys:

        - setMarkersZeroValuesToNaN: if true, marker corrdinates exactly
          matching 0 will be replace with NaNs (e.g. Optitrack systems).
          Default is false.
         
        - removeSegmentNameFromMarkerNames: if true, marker names in the format
        "segment:marker" will be removed the "segment:" prefix.
        Default is false.

    Returns
    -------
    dict
        Collection of read data. It contains, as keys, the items contained
        in ``sections``:

        - markers: this is a dictionary where each key is a point label, and each
          value is a N x 3 np.ndarray of 3D coordinates (in *mm*), where N is the
          number of time frames.

    """

    # Open C3D pointer
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(fileName)
    reader.Update()
    acq = reader.GetOutput()

    # Initialize data structure
    data = {}

    # Read markers
    if 'markers' in sections:

        # Convert points unit to mm
        pointUnit = acq.GetPointUnit()
        if pointUnit == 'mm':
            scaleFactor = 1.
        elif pointUnit == 'm':
            scaleFactor = 1000.
        else:
            raise Exception('Point unit not recognized')

        # Get relevant marker data (N x 3)
        markers = {}
        coll = acq.GetPoints()
        for i in xrange(coll.GetItemNumber()):
            point = coll.GetItem(i)
            label = point.GetLabel()
            if 'removeSegmentNameFromMarkerNames' in opts and opts['removeSegmentNameFromMarkerNames']:
                labelList = label.split(':')
                if len(labelList) > 1:
                    label = labelList[1]
            marker = point.GetValues() * scaleFactor
            if 'setMarkersZeroValuesToNaN' in opts and opts['setMarkersZeroValuesToNaN']:
                marker[marker==0.] = np.nan # replace 0. with np.nan
            markers[label] = marker

        data['markers'] = markers

    # Return data
    return data


def writeC3D(fileName, data, copyFromFile=None):
    """Write to C3D file.

    Parameters
    ----------
    fileName : str
        Full path of the C3D file.

    data : dict
        Data dictionary that can contain the following keys:

        - markers: this is marker-related data. This dictionary contains:
            - data: dictionary where each key is a point label, and each
              value is a N x 3 np.ndarray of 3D coordinates (in *mm*), where N is
              the number of time frames. This field is always necessary.
            - framesNumber: number of data points per marker.
              This field is necessary when creating files from scratch.
            - unit: string indicating the markers measurement unit. Available
              strings are 'mm' and 'm'.
              This field is necessary when creating files from scratch.
            - freq: number indicating the markers acquisition frequency.
              This field is necessary when creating files from scratch.

    copyFromFile : str
        If None, it creates a new file from scratch.
        If str indicating the path of an existing C3D file, it adds/owerwrite data copied from that file.

    """

    if copyFromFile is not None:
        # Open C3D pointer
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(copyFromFile)
        reader.Update()
        acq = reader.GetOutput()
        if 'markers' in data:
            nMarkerFrames = acq.GetPointFrameNumber()
            pointUnit = acq.GetPointUnit()
    else:
        # Create new acquisition
        acq = btk.btkAcquisition()
        if 'markers' in data:
            nMarkerFrames = data['markers']['framesNumber']
            acq.Init(0, nMarkerFrames)
            pointUnit = data['markers']['unit']
            acq.SetPointUnit(pointUnit)
            pointFreq = data['markers']['freq']
            acq.SetPointFrequency(pointFreq)

    if 'markers' in data:
        # Write marker data
        markers = data['markers']['data']
        for m in markers:
            newMarker = btk.btkPoint(m, nMarkerFrames)
            if pointUnit == 'm':
                markerData = markers[m] / 1000.
            elif pointUnit == 'mm':
                markerData = markers[m].copy()
            newMarker.SetValues(markerData)
            acq.AppendPoint(newMarker)

    # Write to C3D
    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(fileName)
    writer.Update()
    

def readMimics(fileName, sections):
    
    # Read file
    with open(fileName) as f:
        content = f.readlines()
        
    # Parse content
    state = 'skip'
    data = {}
    data['markers'] = {}
    for line in content:
        if re.match('Name\s*\t*X1\s*\t*Y1\s*\t*Z1\s*\t*', line):
            state = 'parse'
        elif state == 'parse':
            line = line.strip()
            if line == '':
                state = 'skip'
            else:
                p = line.split('\t')
                name = p[0].strip()
                x = float(p[1])
                y = float(p[2])
                z = float(p[3])
                data['markers'][name] = [x, y, z]
            
    return data

    
def readSTL(filePath):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filePath)
    vtkData = reader.GetOutput()
    return vtkData