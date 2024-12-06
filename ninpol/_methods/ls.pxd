#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp

cnp.import_array()                  # Needed to use NumPy C API

ctypedef long long DTYPE_I_t
ctypedef double DTYPE_F_t

from .._interpolator.logger cimport Logger
from .._interpolator.grid cimport Grid
from .._interpolator.ninpol_defines cimport *

cdef class LSInterpolation:

    cdef readonly int logging
    cdef readonly Logger logger
    cdef dict log_dict


    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data, const DTYPE_F_t[:, ::1] faces_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)
    
    cdef void LS(self, Grid grid,
                 const DTYPE_I_t[::1] points, const DTYPE_I_t[::1] neumann_point,
                 DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)

