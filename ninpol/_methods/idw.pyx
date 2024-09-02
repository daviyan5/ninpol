import numpy as np
from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from libc.stdio cimport printf
from libc.math cimport sqrt

cdef void inverse_distance(const int dim,
                           const DTYPE_F_t[:, ::1] target_coordinates, 
                           const DTYPE_F_t[:, ::1] source_coordinates,
                           const DTYPE_I_t[:, ::1] connectivity_idx,
                           DTYPE_F_t[:, ::1] weights):
    
    cdef int n_target = target_coordinates.shape[0]
    
    cdef:
        int i, j, k
        int source_idx, dest_idx
        int n_source

        int zero_found
        DTYPE_F_t distance = 0.0, total_distance = 0.0
        
        float machine_epsilon = 10 ** int(np.log10(np.finfo(np.float64).eps))
    
    cdef int use_threads = min(8, np.ceil(n_target / 800))
    omp_set_num_threads(use_threads)
    for dest_idx in prange(n_target, nogil=True, schedule='static', num_threads=8 if use_threads else 1):
        zero_found = False
        total_distance = 0
        n_source = 0
        for j, source_idx in enumerate(connectivity_idx[dest_idx]):
            if source_idx == -1:
                break
            
            distance = 0.0
            for k in range(dim):
                distance = distance + (target_coordinates[dest_idx, k] - source_coordinates[source_idx, k])**2
            
            if distance <= machine_epsilon:
                zero_found = True
                for k in range(n_source):
                    weights[dest_idx, k] = 0.
                weights[dest_idx, j] = 1.
                break
            distance = sqrt(distance)

            weights[dest_idx, j] += 1 / distance
            total_distance += 1 / distance

            n_source = n_source + 1

        if not zero_found:
            for k in range(n_source):
                weights[dest_idx, k] /= total_distance
            