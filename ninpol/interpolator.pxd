"""
This file contains the definition of the Interpolator class.
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp
cnp.import_array()                  # Needed to use NumPy C API

from libc.stdio cimport printf
from cython.parallel cimport parallel, prange
cimport openmp

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t

from .grid cimport Grid

cdef class Interpolator:
    cdef Grid grid