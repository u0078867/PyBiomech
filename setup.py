# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 16:33:51 2017

@author: u0078867
"""


from setuptools import setup, find_packages


setup(
    name='PyBiomech',
    version='1.0.0',
    description='Collection of tools for certain biomechanical pipelines',
    long_description=open('README.rst').read(),
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'openpyxl',
        'vtk',
        'btk'
    ],
    include_package_data=True,
    license='MIT',
    author='u0078867',
    author_email='davide.monari@kuleuven.be',
    url='https://github.com/u0078867/PyBiomech',
)
