#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp

cnp.import_array()                  # Needed to use NumPy C API

ctypedef long DTYPE_I_t
ctypedef double DTYPE_F_t


cdef void inverse_distance(const int dim,
                           const DTYPE_F_t[:, ::1] target_coordinates, 
                           const DTYPE_F_t[:, ::1] source_coordinates,
                           const DTYPE_I_t[:, ::1] connectivity_idx,
                           DTYPE_F_t[:, ::1] weights)