# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 16:33:51 2017

@author: u0078867
"""


from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_call


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        print('installing btk ...')
        check_call("easy_install btk".split())
        print('installing vtk ...')
        check_call("conda install vtk=6.3.0".split())


setup(
    name='PyBiomech',
    version='0.33.0',
    description='Collection of tools for certain biomechanical pipelines',
    long_description=open('README.rst').read(),
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=[
        'numpy==1.14.2',
        'scipy==1.1.0',
        'matplotlib==2.1.0',
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    include_package_data=True,
    license='MIT',
    author='u0078867',
    author_email='davide.monari@kuleuven.be',
    url='https://github.com/u0078867/PyBiomech',
)
