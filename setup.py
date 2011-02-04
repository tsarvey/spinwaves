"""
Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant to title 17 section 105* of the United States Code this software is not subject to copyright protection and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. The use of certain trade names or commercial products does not imply any endorsement of a particular product, nor does it imply that the named product is necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise."""


"""This is a setup script to install spinwaves with setup tools.
Author: Tom 9/25/09"""
from os.path import join

#Bootstrap setuptools for systems that do not have it installed
from ez_setup import use_setuptools
use_setuptools()

#For compiling C extensions
#from distutils.core import Extension

from setuptools import setup, find_packages, Extension
setup(
	name = "Spinwaves",
	version = "0.1",
	url = 'http://spinwaves.googlecode.com',
	packages = find_packages(exclude = ['sympy_WORKING']),

	#Add C Extensions
	ext_modules=[Extension(join('spinwaves',join('MonteCarlo','_monteCarlo')), [join('lib',f) for f in ['main1.c','dSFMT.c']])],
	

	entry_points = {'gui_scripts':['spinwaves = spinwaves.vtkModel.wxGUI.GUI_Main:main']},

	install_requires = ['wxPython', 'sympy', 'matplotlib', 'numpy'],

	zip_safe = False
)

#Extension('spinwaves.MonteCarlo._monteCarlo.so', ['main1.c', 'dSFMT.c'], include_dirs=['lib'])
