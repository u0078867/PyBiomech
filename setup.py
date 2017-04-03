# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 16:33:51 2017

@author: u0078867
"""

from setuptools import setup, find_packages

setup(
    name='PyBiomech',
    version='0.0.1.dev1',
    description='Collection of tools for certain biomechanical pipelines',
    long_description='Still in development, use with caution',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'vtk',
        'btk',
    ],
    include_package_data=True,
    license='MIT',
    author='u0078867',
)