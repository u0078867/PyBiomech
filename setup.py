# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 16:33:51 2017

@author: u0078867
"""

#import pypandoc
#long_description = pypandoc.convert('README.md', 'rst', outputfile="README.rst")
#print long_description


from setuptools import setup, find_packages


setup(
    name='PyBiomech',
    version='0.0.6.dev1',
    description='Collection of tools for certain biomechanical pipelines',
    #long_description='Still under development, use with caution!',
    long_description=open('README.rst').read(),
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=[
        'numpy',
        'scipy',
        'vtk',
        'btk',
    ],
    include_package_data=True,
    license='MIT',
    author='u0078867',
    author_email='davide.monari@kuleuven.be',
    url='https://github.com/u0078867/PyBiomech',
)
