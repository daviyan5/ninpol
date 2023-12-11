"""
This file contains the "Grid" class definition, for mesh manipulation. 
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp
cnp.import_array()                  # Needed to use NumPy C API

from libc.stdio cimport printf
from cython.parallel cimport parallel, prange
cimport openmp

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t

cdef class Grid:
    """
    Stores and manipulates the mesh data.
    This is a class intended to be used only from Cython, specifically from the 'interpolator.pyx' file.
    """
    cdef readonly int n_dims             # Number of dimensions
    cdef readonly int n_elems            # Number of elements
    cdef readonly int n_points           # Number of points (vertices)
    cdef readonly int n_points_per_elem  # Number of points per element

    cdef readonly int[:] esup            # Elements surrounding points connectivity
    cdef readonly int[:] esup_ptr        # Elements surrounding points pointer. 
                                         #   i.e: The elements surrounding point i are in esup[esup_ptr[i]:esup_ptr[i+1]]

    cdef readonly int[:] psup            # Points surrounding points connectivity
    cdef readonly int[:] psup_ptr        # Points surrounding points pointer. 
                                         #   i.e: The points surrounding point i are in psup[psup_ptr[i]:psup_ptr[i+1]] 
    
    cpdef void build(self, DTYPE_I_t [:, :] connectivity)