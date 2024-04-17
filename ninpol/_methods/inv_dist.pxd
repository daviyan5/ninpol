#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
cimport numpy as cnp

cnp.import_array()                  # Needed to use NumPy C API

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t


cdef DTYPE_F_t[::1] distance_inverse(int dim, const DTYPE_F_t[:, ::1] target, const DTYPE_F_t[:, ::1] source, 
                                     const DTYPE_I_t[::1] connectivity, const DTYPE_I_t[::1] connectivity_ptr,
                                     int weights_shape, const DTYPE_F_t[::1] weights)