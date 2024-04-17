"""
This file contains the definition of the Interpolator class.
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
from .ninpol_defines cimport *
cimport numpy as cnp
cnp.import_array()                  # Needed to use NumPy C API

from libc.stdio cimport printf
from cython.parallel cimport parallel, prange
cimport openmp

from .grid cimport Grid
from .._methods.inv_dist cimport distance_inverse

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t


cdef class Interpolator:

    cdef readonly dict point_ordering
    cdef readonly dict supported_methods

    cdef readonly object mesh_obj
    cdef readonly Grid grid_obj
    
    cdef readonly dict variable_to_index

    cdef readonly DTYPE_F_t[:, ::1] cells_data 
    cdef readonly DTYPE_I_t[::1] cells_data_dimensions

    cdef readonly DTYPE_F_t[:, ::1] points_data
    cdef readonly DTYPE_I_t[::1] points_data_dimensions
     

    cdef readonly int is_grid_initialized
    
    cdef DTYPE_F_t[::1] inv_dist_interpolator(self, const DTYPE_F_t[::1] source_data, 
                                                  DTYPE_I_t data_dimension, str source_type, 
                                                  const DTYPE_I_t[::1] target)

    

    