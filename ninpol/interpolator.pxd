"""
This file contains the definition of the Interpolator class.
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp
cnp.import_array()                  # Needed to use NumPy C API

from libc.stdio cimport printf
from cython.parallel cimport parallel, prange
cimport openmp
from .mesh cimport grid

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t

import meshio

cdef type MeshioMesh = meshio._mesh.Mesh

cdef class Interpolator:

    cdef readonly dict point_ordering
    cdef readonly object mesh_obj
    cdef readonly grid.Grid grid_obj

    cdef int is_grid_initialized
    
    

    

    