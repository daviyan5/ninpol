#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp

cnp.import_array()                  # Needed to use NumPy C API

ctypedef long long DTYPE_I_t
ctypedef double DTYPE_F_t
from cython cimport view

from .._interpolator.logger cimport Logger
from .._interpolator.pseudo_grid cimport PseudoGrid
from .._interpolator.ninpol_defines cimport *

cdef class LPEW3Interpolation:

    cdef readonly int first_point
    cdef readonly int logging
    cdef readonly Logger logger
    cdef dict log_dict

    cdef readonly double only_dgels

    cdef void prepare(self, PseudoGrid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data, const DTYPE_F_t[:, ::1] faces_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)
    
    cdef void lpew3(self, PseudoGrid grid, const DTYPE_I_t[::1] target_points, DTYPE_F_t[:, :, ::1] permeability, 
                    const DTYPE_I_t[::1] neumann_point, const DTYPE_F_t[::1] neumann_val,
                    DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)

    cdef double partial_lpew3(self, PseudoGrid grid, int point, int elem, DTYPE_F_t[:, :, ::1] permeability)            

    cdef double neumann_treatment(self, PseudoGrid grid, int point, DTYPE_F_t neumann_val)      
         
    cdef double phi_lpew3(self, PseudoGrid grid, int point, int face, int elem, DTYPE_F_t[:, :, ::1] permeability)
    
    cdef double psi_sum_lpew3(self, PseudoGrid grid, int point, int face, int elem, DTYPE_F_t[:, :, ::1] permeability)
    
    cdef double volume(self, int face, DTYPE_I_t[::1] fpoints, DTYPE_F_t[::1] centroid)

    cdef double flux_term(self, DTYPE_F_t[::1] v1, DTYPE_F_t[:, ::1] K, DTYPE_F_t[::1] v2)

    cdef double lambda_lpew3(self, PseudoGrid grid, int point, int aux_point, int face, DTYPE_F_t[:, :, ::1] permeability)
    
    cdef double neta_lpew3(self, PseudoGrid grid, int point, int face, int elem, DTYPE_F_t[:, ::1] K)
    
    cdef double csi_lpew3(self, PseudoGrid grid, int face, int elem, DTYPE_F_t[:, ::1] K)
    
    cdef double sigma_lpew3(self, PseudoGrid grid, int point, int elem, DTYPE_F_t[:, :, ::1] permeability)