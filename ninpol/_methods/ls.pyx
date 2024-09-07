import numpy as np
import scipy as sp

from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
from libc.math cimport sqrt

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

from cython cimport view

DTYPE_I = int
DTYPE_F = float

cdef class LSInterpolation:

    def __cinit__(self, int logging=False):
        self.logging  = logging
        self.log_dict = {}
        self.logger   = Logger("GLS", True)
        self.only_dgels = 0.0


    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        pass
    
    def LS(self, Grid grid,
                 const DTYPE_I_t[::1] points):
        """

        Let w(x, y, z) = a + bx + cy + dz
        Let w_k = w(x_k, y_k, z_k)
        Let Lp = sum((u_k - w_k)**2) be the least squares error
        sum( ( u_k - w_k ) * (nabla_abcd) = 0

        nabla_abc = [1, x_k, y_k, z_k].T

        Thus
        [sum(u_k - w_k), sum(x_k * (u_k - w_k)), sum(y_k * (u_k - w_k)), sum(z_k * (u_k - w_k))] = 0
        
        Finaly, the system is:
        | N  Ix  Iy  Iz  | | a | = | u   |
        | Ix Ixx Ixy Ixz | | b | = | u_x |
        | Iy Ixy Iyy Iyz | | c | = | u_y |
        | Iz Ixz Iyz Izz | | d | = | u_z |

        where N = len(points)
        Ix = sum(x_k), Iy = sum(y_k), Iz = sum(z_k)
        Ixx = sum(x_k**2), Ixy = sum(x_k * y_k), Ixz = sum(x_k * z_k)
        Iyy = sum(y_k**2), Iyz = sum(y_k * z_k), Izz = sum(z_k**2)
        u = sum(u_k), u_x = sum(x_k * u_k), u_y = sum(y_k * u_k), u_z = sum(z_k * u_k)

        Thus:
            weights[point, k] = w_k
            Where
            w_k = (1 + lambda_x * x_k + lambda_y * y_k + lambda_z * z_k) / (n_k + lambda_x * Ix + lambda_y * Iy + lambda_z * Iz)

            lambda_x = (I_xy * I_y + I_xz * I_z - I_yy * I_x - I_zz * I_z) / D
        """
        for point in points:
            pass
            