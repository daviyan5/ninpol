#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas

cnp.import_array()                  # Needed to use NumPy C API

ctypedef long DTYPE_I_t
ctypedef double DTYPE_F_t
from cython cimport view

from .._interpolator.logger cimport Logger
from .._interpolator.grid cimport Grid
from .._interpolator.ninpol_defines cimport *

cdef class LSInterpolation:

    cdef readonly int logging
    cdef readonly Logger logger
    cdef dict log_dict


    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws)

