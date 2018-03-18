from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'tsp_lib_cython',
  ext_modules = cythonize("tsplib_Cy.pyx"),
  include_dirs=[numpy.get_include()]

)

