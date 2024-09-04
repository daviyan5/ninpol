#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas

cnp.import_array()                  # Needed to use NumPy C API

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t
from cython cimport view

from .._interpolator.logger cimport Logger
from .._interpolator.grid cimport Grid
from .._interpolator.ninpol_defines cimport *

cdef class GLSInterpolation:

    cdef readonly int logging
    cdef readonly Logger logger
    cdef dict log_dict
    cdef readonly Grid grid_obj

    cdef void GLS(self, Grid grid, const DTYPE_I_t[::1] points, 
                  DTYPE_F_t[:, :, ::1] permeability, 
                  const DTYPE_F_t[::1] diff_mag, 
                  const DTYPE_I_t[::1] neumann_point, const DTYPE_F_t[::1] neumann_val,
                  DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)

    cdef view.array array(self, tuple shape, str t)
    cdef DTYPE_F_t[::1] cross(self, const DTYPE_F_t[::1] a, const DTYPE_F_t[::1] b)
    cdef DTYPE_F_t norm(self, const DTYPE_F_t[::1] a)
    
    
    cdef void build_ks_sv_arrays(self, Grid grid, int point, 
                                 DTYPE_I_t[::1] KSetv, DTYPE_I_t[::1] Sv, DTYPE_I_t[::1] Svb)

    cdef void build_ls_matrices(self, Grid grid, int point, 
                                const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb,
                                DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] diff_mag,
                                DTYPE_F_t[:, ::1] Mi, DTYPE_F_t[:, ::1] Ni)
    
    cdef void _set_mi(self, 
                     const int row, const int col, 
                     const DTYPE_F_t[::1] v, DTYPE_F_t[:, ::1] Mi, int k)

    cdef void set_neumann_rows(self, Grid grid,
                               int point, const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb,
                               DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] neumann_val,
                               DTYPE_F_t[:, ::1] Mi, DTYPE_F_t[:, ::1] Ni)

    cdef void solve_ls(self, int point, int is_neumann,
                       DTYPE_F_t[:, ::1] Mi, DTYPE_F_t[:, ::1] Ni, 
                       DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)
