"""
This file contains the definition of the Interpolator class.
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
from .ninpol_defines cimport *
cimport numpy as cnp
cnp.import_array()                  # Needed to use NumPy C API

from libc.stdio cimport printf
from cython.parallel import prange
cimport openmp

from .grid cimport Grid
from .logger cimport Logger

from .._methods.idw cimport IDWInterpolation
from .._methods.gls cimport GLSInterpolation
from .._methods.ls  cimport LSInterpolation

ctypedef long DTYPE_I_t
ctypedef double DTYPE_F_t


cdef class Interpolator:

    cdef readonly dict point_ordering
    cdef readonly dict supported_methods
    cdef readonly dict types_per_dimension

    cdef readonly object mesh_obj
    cdef readonly Grid grid

    cdef readonly GLSInterpolation gls
    cdef readonly IDWInterpolation idw
    cdef readonly LSInterpolation ls
    
    cdef readonly dict variable_to_index

    cdef readonly DTYPE_F_t[:, ::1] cells_data
    cdef readonly DTYPE_I_t[::1] cells_data_dimensions

    cdef readonly DTYPE_F_t[:, ::1] points_data
    cdef readonly DTYPE_I_t[::1] points_data_dimensions

    cdef readonly DTYPE_F_t[:, ::1] faces_data
    cdef readonly DTYPE_I_t[::1] faces_data_dimensions
    
    cdef readonly int logging
    cdef readonly Logger logger

    cdef readonly int build_edges

    cdef readonly int is_grid_initialized

    cdef tuple process_mesh(self, object mesh)
    cpdef void load_mesh(self, str filename = *, object mesh_obj = *)

    cdef void load_cell_data(self)
    cdef void load_point_data(self)
    cpdef void load_face_data(self, dict data_dict, DTYPE_I_t[:, ::1] face_connectivity = *)
    cdef void load_data(self, dict data_dict, str data_type)

    cdef DTYPE_F_t[::1] compute_diffusion_magnitude(self, DTYPE_F_t[:, ::1] permeability)

    cpdef tuple interpolate(self, str variable, str method, DTYPE_I_t[::1] target_points = *)
    
    cdef tuple prepare_interpolator(self, str method, str variable,
                                    const DTYPE_I_t[::1] target_points)

    

    