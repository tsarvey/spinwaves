"""This is a Cython setup file to compile the McQueeny_C_alg file."""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("mcQueeny_C_alg", ["mcQueeny_C_alg.pyx"])]

setup(
    name = 'McQueeny Cross Section Algorithm',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs = [np.get_include()]
)