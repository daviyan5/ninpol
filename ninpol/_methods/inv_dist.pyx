import numpy as np
from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from libc.stdio cimport printf
from libc.math cimport sqrt

cdef DTYPE_F_t[::1] distance_inverse(int dim, const DTYPE_F_t[:, ::1] target, const DTYPE_F_t[:, ::1] source, 
                                     const DTYPE_I_t[::1] connectivity, const DTYPE_I_t[::1] connectivity_ptr,
                                     int weights_shape, const DTYPE_F_t[::1] weights):
    
    cdef int n_target = target.shape[0]


    cdef DTYPE_F_t[::1] result = np.zeros(n_target * weights_shape, dtype=np.float64)
    cdef:
        int i, j, k
        int source_idx, dest_idx
        int zero_found
        DTYPE_F_t distance = 0.0, total_distance = 0.0
        int use_threads = n_target > 1000
        float machine_epsilon = 10 ** int(np.log10(np.finfo(np.float64).eps))
    
    omp_set_num_threads(8 if use_threads else 1)
    for dest_idx in prange(n_target, nogil=True, schedule='static', num_threads=8 if use_threads else 1):
        total_distance = 0
        zero_found = False

        for j in range(connectivity_ptr[dest_idx], connectivity_ptr[dest_idx+1]):
            source_idx = connectivity[j]
            distance = 0.0
            for k in range(dim):
                distance = distance + (target[dest_idx, k] - source[source_idx, k])**2
            
            if distance <= machine_epsilon:
                zero_found = True
                for k in range(weights_shape):
                    result[dest_idx * weights_shape + k] = weights[source_idx * weights_shape + k]
                break 
            distance = sqrt(distance)
            for k in range(weights_shape):
                result[dest_idx * weights_shape + k] += weights[source_idx * weights_shape + k] * (1 / distance)
            total_distance = total_distance + 1 / distance

        if not zero_found:
            for k in range(weights_shape):
                result[dest_idx * weights_shape + k] /= total_distance
        else:
            zero_found = False
        
    
    return result

