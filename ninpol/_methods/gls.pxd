#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp


cnp.import_array()                  # Needed to use NumPy C API

ctypedef long DTYPE_I_t
ctypedef double DTYPE_F_t
from cython cimport view

from .._interpolator.logger cimport Logger
from .._interpolator.grid cimport Grid
from .._interpolator.ninpol_defines cimport *

cdef class GLSInterpolation:

    cdef readonly int first_point
    cdef readonly int logging
    cdef readonly Logger logger
    cdef dict log_dict

    cdef readonly double only_dgels

    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data, const DTYPE_F_t[:, ::1] faces_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)

    cdef void GLS(self, Grid grid, const DTYPE_I_t[::1] points, 
                  DTYPE_F_t[:, :, ::1] permeability, 
                  const DTYPE_F_t[::1] diff_mag, 
                  const DTYPE_I_t[::1] neumann_point, const DTYPE_F_t[::1] neumann_val,
                  DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)

    cdef view.array array(self, tuple shape, str t)
    cdef void cross(self, const DTYPE_F_t[::1] a, const DTYPE_F_t[::1] b, DTYPE_F_t[::1] c) noexcept nogil 
    cdef DTYPE_F_t norm(self, const DTYPE_F_t[::1] a) noexcept nogil
    
    
    cdef void build_ks_sv_arrays(self, Grid grid, int point, 
                                 DTYPE_I_t[::1] KSetv, DTYPE_I_t[::1] Sv, DTYPE_I_t[::1] Svb, 
                                 const int n_elem, const int n_face, const int n_bface) noexcept nogil

    cdef void build_ls_matrices(self, Grid grid, int point, 
                                const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb, 
                                const int n_elem, const int n_face, const int n_bface, 
                                DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] diff_mag,
                                DTYPE_F_t[::1] xv, DTYPE_F_t[:, ::1] xK, DTYPE_F_t[:, ::1] dKv,
                                DTYPE_F_t[:, ::1] xS, DTYPE_F_t[:, ::1] N_sj, DTYPE_I_t[:, ::1] Ks_Sv, DTYPE_F_t[::1] eta_j,
                                DTYPE_F_t[:, ::1] T_sj1, DTYPE_F_t[:, ::1] T_sj2, DTYPE_F_t[::1] tau_j2, DTYPE_F_t[:, ::1] tau_tsj2,
                                DTYPE_F_t[:, ::1] nL1, DTYPE_F_t[:, ::1] nL2, DTYPE_I_t[::1] Ij1, DTYPE_I_t[::1] Ij2, DTYPE_F_t[::1] temp_cross,
                                DTYPE_I_t[::1] idx1, DTYPE_I_t[::1] idx2, DTYPE_I_t[::1] idx3,
                                DTYPE_F_t[:, ::1] Mi, DTYPE_F_t[:, ::1] Ni) noexcept nogil
    
    cdef void _set_mi(self, 
                     const int row, const int col, 
                     const DTYPE_F_t[::1] v, DTYPE_F_t[:, ::1] Mi, int k) noexcept nogil

    cdef void set_neumann_rows(self, Grid grid,
                               int point, const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb, 
                               const int n_elem, const int n_face, const int n_bface, 
                               DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] neumann_val,
                               DTYPE_I_t[::1] neumann_rows, DTYPE_I_t[:, ::1] Ks_Svb, 
                               DTYPE_F_t[:, ::1] nL, DTYPE_I_t[::1] Ik,
                               DTYPE_F_t[:, ::1] Mi, DTYPE_F_t[:, ::1] Ni) noexcept nogil

    cdef void solve_ls(self, int point, int is_neumann,
                       DTYPE_F_t[:, ::1] Mi, DTYPE_F_t[:, ::1] Ni, 
                       int m, int n, int nrhs,
                       int lda, int ldb,
                       DTYPE_F_t[::1] work, int lwork,
                       DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws) noexcept nogil
