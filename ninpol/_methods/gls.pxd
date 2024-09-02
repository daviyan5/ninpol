#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp

cnp.import_array()                  # Needed to use NumPy C API

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t

from .._interpolator.logger cimport Logger
from .._interpolator.grid cimport Grid

cdef void GLS(Grid grid, 
              const DTYPE_I_t[::1] in_points, 
              const DTYPE_I_t[::1] nm_points, 
              const DTYPE_F_t[:, :, ::1] permeability, 
              const DTYPE_F_t[::1] diff_mag,
              const DTYPE_F_t[::1] neumann,
              DTYPE_F_t[:, ::1] weights, 
              DTYPE_F_t[::1] neumann_ws)