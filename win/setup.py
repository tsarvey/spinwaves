"""
Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant to title 17 section 105* of the United States Code this software is not subject to copyright protection and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. The use of certain trade names or commercial products does not imply any endorsement of a particular product, nor does it imply that the named product is necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise."""


from distutils.core import setup
import glob
import py2exe
import matplotlib
import sys
import os

import sys
import os
#Add the main folder (up one level) to the path so 'spinwaves' can be imported
spinwaves_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(spinwaves_path)

#This is used by py2exe to create a windows executable
#In DOS prompt:
#python setup.py py2exe



#python_dir = "C:\Python25"
python_dir=os.path.dirname(sys.executable)
#Matplotlib code taken from: http://www.py2exe.org/index.cgi/MatPlotLib

# We need to exclude matplotlib backends not being used by this executable.  You may find
# that you need different excludes to create a working executable with your chosen backend.
# We also need to include include various numerix libraries that the other functions call.

opts = {
    'py2exe': { "includes" : ["sip", "matplotlib.backends",  "matplotlib.backends.backend_qt4agg", "matplotlib.figure",
                              "matplotlib.backends.backend_wxagg", "pylab", "numpy", "matplotlib.numerix.fft",
                              "matplotlib.numerix.linear_algebra", "matplotlib.numerix.random_array",
                              "matplotlib.backends.backend_tkagg"],
                'excludes': ['_gtkagg', '_tkagg', '_agg2', '_cairo', '_cocoaagg',
                             '_fltkagg', '_gtk', '_gtkcairo', ],
                'dll_excludes': ['libgdk-win32-2.0-0.dll',
                                 'libgobject-2.0-0.dll'],
                "compressed": 1,
                "optimize": 0,
                "bundle_files":3,
                'typelibs' : [('{EAB22AC0-30C1-11CF-A7EB-0000C05BAE0B}', 0, 1, 1)],
              }
       }
 
# Save matplotlib-data to mpl-data ( It is located in the matplotlib\mpl-data
# folder and the compiled programs will look for it in \mpl-data
# note: using matplotlib.get_mpldata_info
#data_files = [(r'mpl-data', glob.glob(python_dir + r'\Lib\site-packages\matplotlib\mpl-data\*.*')),
#                    # Because matplotlibrc does not have an extension, glob does not find it (at least I think that's why)
#                    # So add it manually here:
#                  (r'mpl-data', [python_dir + r'\Lib\site-packages\matplotlib\mpl-data\matplotlibrc']),
#                  (r'mpl-data\images',glob.glob(python_dir + r'\Lib\site-packages\matplotlib\mpl-data\images\*.*')),
#                  (r'mpl-data\fonts',glob.glob(python_dir + r'\Lib\site-packages\matplotlib\mpl-data\fonts\*.*'))]
#for inno setup
data_files=matplotlib.get_py2exe_datafiles()
data_files.append("screen.ico")
data_files.append("..\spinwaves\MonteCarlo\_monteCarlo.pyd")
setup(windows=[{"script" : "Spinwaves.py", "icon_resources": [(0x0004, "screen.ico")]}], 
      console=[],
      options=opts,   
      data_files=data_files
      )






