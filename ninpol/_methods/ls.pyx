import numpy as np

from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from openmp cimport omp_init_lock, omp_destroy_lock, omp_set_lock, omp_unset_lock, omp_lock_t
from libc.stdio cimport printf
from libc.math cimport sqrt

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

DTYPE_I = int
DTYPE_F = float

cdef class LSInterpolation:

    def __cinit__(self, int logging=False):
        self.logging  = logging
        self.log_dict = {}
        self.logger   = Logger("GLS")


    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data, const DTYPE_F_t[:, ::1] faces_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        cdef:
            int neumann_flag_index       = variable_to_index["points"]["neumann_flag" + "_" + variable]
            DTYPE_I_t[::1] neumann_point = np.asarray(points_data[neumann_flag_index]).astype(DTYPE_I)

        self.LS(grid, target_points, neumann_point, weights, neumann_ws)

    cdef void LS(self, Grid grid,
                 const DTYPE_I_t[::1] points, const DTYPE_I_t[::1] neumann_point,
                 DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        cdef:
            double Ix, Iy, Iz
            double Ixx, Ixy, Ixz, Iyy, Iyz, Izz
            double D
            double lambda_x, lambda_y, lambda_z
            double denom

            double volx, voly, volz

            int n_vols
            int point, vol, i, idx

            double total_distance = 0.0


            int n_target = points.shape[0]

            int use_threads = min(16, np.ceil(n_target / 400))

        omp_set_num_threads(use_threads)
        for idx in prange(n_target, nogil=True, schedule='static', num_threads=use_threads):   
            point = points[idx]
            if grid.boundary_points[point] and not neumann_point[point]: 
                continue
            Ix = Iy = Iz = 0.0
            Ixx = Ixy = Ixz = Iyy = Iyz = Izz = 0.0
            n_vols = grid.esup_ptr[point + 1] - grid.esup_ptr[point]

            for vol in grid.esup[grid.esup_ptr[point]:grid.esup_ptr[point + 1]]:
                volx = grid.centroids[vol, 0] - grid.point_coords[point, 0]
                voly = grid.centroids[vol, 1] - grid.point_coords[point, 1]
                volz = grid.centroids[vol, 2] - grid.point_coords[point, 2]

                Ix  = Ix  + volx
                Iy  = Iy  + voly
                Iz  = Iz  + volz
                Ixx = Ixx + volx * volx
                Ixy = Ixy + volx * voly
                Ixz = Ixz + volx * volz
                Iyy = Iyy + voly * voly
                Iyz = Iyz + voly * volz
                Izz = Izz + volz * volz

            if Iz == 0.0 and Izz == 0.0 and Ixz == 0.0 and Iyz == 0.0:
                Izz = 1.0

            D = (
                Ixx * (Iyy * Izz - Iyz * Iyz) +
                Ixy * (Iyz * Ixz - Ixy * Izz) +
                Ixz * (Ixy * Iyz - Iyy * Ixz)
            )

            if D == 0.0:
                # Use inverse distance weighting to deal with this corner case
                total_distance = 0.0
                for i, vol in enumerate(grid.esup[grid.esup_ptr[point]:grid.esup_ptr[point + 1]]):
                    volx = grid.centroids[vol, 0] - grid.point_coords[point, 0]
                    voly = grid.centroids[vol, 1] - grid.point_coords[point, 1]
                    volz = grid.centroids[vol, 2] - grid.point_coords[point, 2]

                    weights[point, i] = 1.0 / sqrt(volx * volx + voly * voly + volz * volz)
                    total_distance = total_distance + 1.0 / sqrt(volx * volx + voly * voly + volz * volz)

                for i, vol in enumerate(grid.esup[grid.esup_ptr[point]:grid.esup_ptr[point + 1]]):
                    weights[point, i] = weights[point, i] / total_distance
                
                continue
                    

            if Iz == 0.0 and Izz == 0.0 and Ixz == 0.0 and Iyz == 0.0:
                Izz = -1.0
            
            lambda_x = ( 
                Ix * (Iyz * Iyz - Iyy * Izz) +
                Iy * (Ixy * Izz - Iyz * Ixz) +
                Iz * (Iyy * Ixz - Ixy * Iyz)
            ) / D
            
            lambda_y = (
                Ix * (Ixy * Izz - Iyz * Ixz) +
                Iy * (Ixz * Ixz - Ixx * Izz) +
                Iz * (Ixx * Iyz - Ixy * Ixz)
            ) / D
            
            lambda_z = (
                Ix * (Iyy * Ixz - Ixy * Iyz) + 
                Iy * (Ixx * Iyz - Ixy * Ixz) +
                Iz * (Ixy * Ixy - Ixx * Iyy)
            ) / D
            
            denom = n_vols + lambda_x * Ix + lambda_y * Iy + lambda_z * Iz
            for i, vol in enumerate(grid.esup[grid.esup_ptr[point]:grid.esup_ptr[point + 1]]):
                volx = grid.centroids[vol, 0] - grid.point_coords[point, 0]
                voly = grid.centroids[vol, 1] - grid.point_coords[point, 1]
                volz = grid.centroids[vol, 2] - grid.point_coords[point, 2]
                weights[point, i] = (
                    (1. + lambda_x * volx + 
                          lambda_y * voly + 
                          lambda_z * volz)
                )
                weights[point, i] /= denom