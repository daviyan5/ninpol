#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp

cnp.import_array()                  # Needed to use NumPy C API

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t


cdef void distance_inverse(const int dim,
                           const DTYPE_F_t[:, ::1] target_coordinates, 
                           const DTYPE_F_t[:, ::1] source_coordinates,
                           const DTYPE_I_t[:, ::1] connectivity_idx,
                           DTYPE_F_t[:, ::1] weights)