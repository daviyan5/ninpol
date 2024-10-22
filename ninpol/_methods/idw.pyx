import numpy as np
from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from libc.stdio cimport printf
from libc.math cimport sqrt

cdef class IDWInterpolation:

    def __cinit__(self, int logging=False):
        self.logging = logging
        self.logger  = Logger("IDW")
        self.log_dict = {}

    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data, const DTYPE_F_t[:, ::1] faces_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        cdef:
            int dim = grid.dim
            int n_target = target_points.shape[0]

            DTYPE_F_t[:, ::1] target_coordinates = np.asarray(grid.point_coords)[target_points]
            DTYPE_F_t[:, ::1] source_coordinates = np.asarray(grid.centroids)

            int neumann_flag_index = variable_to_index["points"]["neumann_flag" + "_" + variable]
            const DTYPE_I_t[::1] neumann_point   = np.asarray(points_data[neumann_flag_index]).astype(int)

        self.inverse_distance(dim, grid, target_points, target_coordinates, source_coordinates, neumann_point, weights)
        
            
            

    cdef void inverse_distance(self, const int dim, Grid grid,
                               const DTYPE_I_t[::1] target_points,
                               const DTYPE_F_t[:, ::1] target_coordinates, 
                               const DTYPE_F_t[:, ::1] source_coordinates,
                               const DTYPE_I_t[::1] neumann_point,
                               DTYPE_F_t[:, ::1] weights):
        
        cdef int n_target = target_coordinates.shape[0]
        
        cdef:
            int i, j, k
            int source_idx, dest_idx
            int n_source
            int point

            int zero_found
            DTYPE_F_t distance = 0.0, total_distance = 0.0
            
            float machine_epsilon = 10 ** int(np.log10(np.finfo(np.float64).eps))
        
        cdef int use_threads = min(16, np.ceil(n_target / 400))
        omp_set_num_threads(use_threads)
        for dest_idx in prange(n_target, nogil=True, schedule='static', num_threads=use_threads):
            point = target_points[dest_idx]
            zero_found = False
            total_distance = 0
            n_source = 0
            if grid.boundary_points[point] and not neumann_point[point]:
                continue
            for j, source_idx in enumerate(grid.esup[grid.esup_ptr[point]:grid.esup_ptr[point + 1]]):
                distance = 0.0
                for k in range(dim):
                    distance = distance + (target_coordinates[dest_idx, k] - source_coordinates[source_idx, k])**2
                
                if distance <= machine_epsilon:
                    zero_found = True
                    for k in range(n_source):
                        weights[point, k] = 0.
                    weights[point, j] = 1.
                    break
                distance = sqrt(distance)

                weights[point, j] += 1 / distance
                total_distance += 1 / distance

                n_source = n_source + 1

            if not zero_found:
                for k in range(n_source):
                    weights[point, k] /= total_distance
            