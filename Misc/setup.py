from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'tsp_lib_cython',
  ext_modules = cythonize("Fibonacci_Cy.pyx"),
)