# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 10:55:54 2017

@author: u0078867
"""

import sys
import os

modPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.append(modPath)

from PyBiomech import io

data = io.readMimics('landmarks.mimics.txt', [])
print data