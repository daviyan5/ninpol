"""
This file contains the "Grid" class definition, for mesh manipulation. 
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
from .ninpol_defines cimport *
from .logger cimport Logger
cimport numpy as cnp
cnp.import_array()                  # Needed to use NumPy C API

ctypedef long long DTYPE_I_t
ctypedef double  DTYPE_F_t

cdef class PseudoGrid:
    cdef int logging
    cdef Logger logger

    cdef int build_edges
    
    cdef readonly int dim

    cdef readonly int n_elems
    cdef readonly int n_points
    cdef readonly int n_faces
    cdef readonly int n_edges

    cdef readonly int are_elements_loaded
    cdef readonly int are_coords_loaded
    cdef readonly int are_structures_built
    cdef readonly int are_centroids_calculated
    cdef readonly int are_normals_calculated
    
    cdef readonly int MX_ELEMENTS_PER_POINT
    cdef readonly int MX_POINTS_PER_POINT
    cdef readonly int MX_ELEMENTS_PER_FACE
    cdef readonly int MX_FACES_PER_POINT
    

    cdef readonly DTYPE_F_t[:, ::1] point_coords
    cdef readonly DTYPE_F_t[:, ::1] centroids
    cdef readonly DTYPE_F_t[:, ::1] normal_faces
    cdef readonly DTYPE_F_t[:, ::1] faces_centers
    cdef readonly DTYPE_F_t[::1] faces_areas
    
    cdef readonly DTYPE_I_t[::1] boundary_faces

    cdef readonly DTYPE_I_t[::1] boundary_points
    
    cdef DTYPE_I_t[::1] element_types

    cdef DTYPE_I_t[::1] npoel
    cdef readonly DTYPE_I_t[:, ::1] inpoel

    cdef readonly DTYPE_I_t[::1] esup
    cdef readonly DTYPE_I_t[::1] esup_ptr

    cdef readonly DTYPE_I_t[::1] psup
    cdef readonly DTYPE_I_t[::1] psup_ptr

    cdef readonly DTYPE_I_t[::1] fsup
    cdef readonly DTYPE_I_t[::1] fsup_ptr

    cdef DTYPE_I_t[::1] nfael
    cdef DTYPE_I_t[:, ::1] lnofa
    cdef DTYPE_I_t[:, :, ::1] lpofa

    cdef readonly DTYPE_I_t[:, ::1] inpofa
    cdef readonly DTYPE_I_t[:, ::1] infael

    cdef readonly DTYPE_I_t[::1] esuf
    cdef readonly DTYPE_I_t[::1] esuf_ptr

    cdef readonly DTYPE_I_t[:, ::1] esuel

    cdef DTYPE_I_t[::1] nedel
    cdef DTYPE_I_t[:, :, ::1] lpoed

    cdef readonly DTYPE_I_t[:, ::1] inpoed
    cdef readonly DTYPE_I_t[:, ::1] inedel