"""
Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant to title 17 section 105* of the United States Code this software is not subject to copyright protection and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. The use of certain trade names or commercial products does not imply any endorsement of a particular product, nor does it imply that the named product is necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise."""


"""
    1D Modeling for SANS
"""
## \mainpage Analytical Modeling for SANS
#
# \section intro_sec Introduction
# This module provides theoretical models for the scattering 
# intensity for SANS. 
#
# Documentation can be found here: 
#    http://danse.us/trac/sans/wiki/8_2_2_1DModelFitting
#    http://danse.us/trac/sans/wiki/8_2_3_2DModeling
#
# \section install_sec Installation
#
# \subsection obtain Obtaining the Code
#
# The code is available here:
# \verbatim
#$ svn co svn://danse.us/sans/sansmodels
# \endverbatim
#
# \subsection depends External Dependencies
# None
#
# \subsection build Building the code
# The standard python package can be built with distutils.
# \verbatim
#$ python setup.py build
#$ python setup.py install
# \endverbatim
#
# \section overview_sec Package Overview
# 
# \subsection class Class Diagram:
# \image html class_diag.png
# Note that the CCylinderModel is written as C code. 
# CylinderModel acts as an adaptor class for the C extension.
# Most model classes will be written that way.
#
# \subsection behav Behavior enumeration for pyre-level architecture:
# \image html behavior_pyre.png
#
# \subsection behav Behavior enumeration for under-lying architecture:
# \image html behavior.jpg
#
# \subsection Tutorial
# To create a model:
# \verbatim
#from sans.models.ModelFactory import ModelFactory
#    cyl = ModelFactory().getModel('CylinderModel')
# \endverbatim
#
# To evaluate a model (at x=0.1 in this example):
# \verbatim
#    output = cyl.run(0.1)
# \endverbatim
#
# To change a parameter:
# \verbatim
#    cyl.setParam('scale', 0.1)
# \endverbatim
#
# To get the value of a parameter:
# \verbatim
#    cyl.getParam('scale')
# \endverbatim
#
# Other examples are available as unit tests under sans.models.test.
#
# \section help_sec Contact Info
# Code and Documentation by Mathieu Doucet as part of the DANSE project.

__author__ = 'Mathieu Doucet'
