from setuptools import setup

""" Setup.py is used to install the package.
-----------------------------------------------
    name : is the name of the package.
    version : is the versioning of the package.
    description: add short description of the package.
    long_description: long description of the package.
    author: Add Author Names.
    packages: add all the packages[folders] in the package.
    install_requires: add all the immediate requirements to the package.
    Install the package by running : python setup.py install.
"""

setup(
    name="zexture",
    version="0.1",
    description="Zexture: gesture recognition module",
    long_description="""Jesture is openCV based hand gesture detection module.
            It uses the mediapipe library to detect hands by returning a set of 21 landmarks for each camera frame which contains a hand.""",
    author="auddy99, singh2010nidhi, mayankraj",
    packages=['zexture'],
    install_requires=[
        'opencv-contrib-python==4.5.5.62',
        'mediapipe==0.8.9.1',
        'numpy==1.22.2',
        'pandas==1.4.1',
        'sklearn==0.0'
        ]
)




