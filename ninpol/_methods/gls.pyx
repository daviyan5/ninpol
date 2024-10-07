import numpy as np
from scipy.__config__ import CONFIG
import os

from ..utils.robin_hood cimport unordered_map
from libc.math cimport sqrt

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

from cython cimport view

DTYPE_I = int
DTYPE_F = float

cdef class GLSInterpolation:
    def __cinit__(self, int logging=False):
        self.logging  = logging
        self.log_dict = {}
        self.logger   = Logger("GLS", False, ".gls_log")
        self.only_dgels = 0.0

    cdef void prepare(self, Grid grid, 
                      const DTYPE_F_t[:, ::1] cells_data, const DTYPE_F_t[:, ::1] points_data, const DTYPE_F_t[:, ::1] faces_data,
                      dict variable_to_index,
                      str variable,
                      const DTYPE_I_t[::1] target_points,
                      DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        
        cdef:
            int dim = grid.dim
            int permeability_index = variable_to_index["cells"]["permeability"]
            int diff_mag_index     = variable_to_index["cells"]["diff_mag"]
            int neumann_flag_index = variable_to_index["points"]["neumann_flag" + "_" + variable]
            int neumann_val_index  = variable_to_index["points"]["neumann" + "_" + variable]

            DTYPE_F_t[:, :, ::1] permeability  = np.reshape(cells_data[permeability_index], 
                                                            (grid.n_elems, dim, dim))
            
            const DTYPE_F_t[::1] diff_mag      = cells_data[diff_mag_index]

            const DTYPE_I_t[::1] neumann_point = np.asarray(points_data[neumann_flag_index]).astype(DTYPE_I)

            const DTYPE_F_t[::1] neumann_val   = points_data[neumann_val_index]

            cdef str lapack_env = "OPENBLAS_NUM_THREADS"        # Assumes OpenBLAS is used
            cdef str scipy_lapack = "scipy-openblas"
            cdef dict scipy_lapack_to_env = {
                "scipy-openblas": "OPENBLAS_NUM_THREADS",
                "scipy-atlas": "ATLAS_NUM_THREADS",
                "scipy-mkl": "MKL_NUM_THREADS"
            }
        # Check if scipy is using OpenBLAS, ATLAS or MKL
        try:
            scipy_lapack = CONFIG['Build Dependencies']['lapack']['name']
            lapack_env = scipy_lapack_to_env[scipy_lapack]
        except KeyError:
            pass
        
        cdef str previous_lapack_env = os.environ.get(lapack_env, "1")
        os.environ[lapack_env] = "1"
        self.GLS(grid, target_points, permeability, diff_mag, neumann_point, neumann_val, weights, neumann_ws)
        os.environ[lapack_env] = previous_lapack_env

        
    cdef void GLS(self, Grid grid, const DTYPE_I_t[::1] points, 
                  DTYPE_F_t[:, :, ::1] permeability, 
                  const DTYPE_F_t[::1] diff_mag, 
                  const DTYPE_I_t[::1] neumann_point, const DTYPE_F_t[::1] neumann_val,
                  DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
    
        cdef:
            int point
            int n_points = points.shape[0]

        cdef:
            int n_elem
            int n_face
            int face
            int n_bface
        
        cdef:
            int N_ELEM_MAX  = grid.MX_ELEMENTS_PER_POINT
            int N_FACE_MAX  = grid.MX_FACES_PER_POINT
            int N_BFACE_MAX = grid.MX_FACES_PER_POINT 

            int m    = N_ELEM_MAX + 3 * N_FACE_MAX + N_BFACE_MAX
            int n    = 3 * N_ELEM_MAX + 1
            int nrhs = N_ELEM_MAX + 1

            int M_MAX = m
            int N_MAX = n
            int NRHS_MAX = nrhs

            int lda = max(1, m)
            int ldb = max(1, m)
        
        cdef:
            DTYPE_I_t[::1] KSetv = np.zeros(N_ELEM_MAX, dtype=DTYPE_I)
            DTYPE_I_t[::1] Sv    = np.zeros(N_FACE_MAX, dtype=DTYPE_I)
            DTYPE_I_t[::1] Svb   = np.zeros(N_BFACE_MAX, dtype=DTYPE_I)

        cdef:
            DTYPE_F_t[::1, :] Mi = view.array(shape=(m, n), itemsize=sizeof(double), format="d", mode='fortran')
            DTYPE_F_t[::1, :] Ni = view.array(shape=(m, nrhs), itemsize=sizeof(double), format="d", mode='fortran')

        cdef:
            DTYPE_F_t[:, ::1] xS        = np.zeros((N_FACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[::1] xv           = np.zeros((3,           ), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] xK        = np.zeros((N_ELEM_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] dKv       = np.zeros((N_ELEM_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] N_sj      = np.zeros((N_FACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_I_t[:, ::1] Ks_Sv     = np.zeros((N_FACE_MAX, 2), dtype=DTYPE_I)
            DTYPE_F_t[::1] eta_j        = np.zeros((N_FACE_MAX,  ), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] T_sj1     = np.zeros((N_FACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] T_sj2     = np.zeros((N_FACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[::1] tau_j2       = np.zeros((N_FACE_MAX,  ), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] tau_tsj2  = np.zeros((N_FACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] nL1       = np.zeros((N_FACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[:, ::1] nL2       = np.zeros((N_FACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_F_t[::1] temp_cross   = np.zeros((3,           ), dtype=DTYPE_F)

        cdef:
            DTYPE_I_t[::1] Ij1  = np.zeros(N_FACE_MAX, dtype=DTYPE_I)
            DTYPE_I_t[::1] Ij2  = np.zeros(N_FACE_MAX, dtype=DTYPE_I)
            DTYPE_I_t[::1] idx1 = np.zeros(N_FACE_MAX, dtype=DTYPE_I)
            DTYPE_I_t[::1] idx2 = np.zeros(N_FACE_MAX, dtype=DTYPE_I)
            DTYPE_I_t[::1] idx3 = np.zeros(N_FACE_MAX, dtype=DTYPE_I)

        cdef:
            DTYPE_I_t[::1] neumann_rows = np.zeros(N_BFACE_MAX, dtype=DTYPE_I)
            DTYPE_I_t[:, ::1] Ks_Svb    = np.zeros((N_BFACE_MAX, 1), dtype=DTYPE_I)
            DTYPE_F_t[:, ::1] nL        = np.zeros((N_BFACE_MAX, 3), dtype=DTYPE_F)
            DTYPE_I_t[::1] Ik           = np.zeros(N_BFACE_MAX, dtype=DTYPE_I)

        cdef:
            double[::1] work = np.zeros(1, dtype=DTYPE_F)
            
            int lwork = -1
            int info = 0

        lapack.dgels('N', &m, &n, &nrhs, &Mi[0, 0], &lda, &Ni[0, 0], &ldb, &work[0], &lwork, &info)
        lwork = int(work[0])
        work = np.zeros(lwork, dtype=DTYPE_F)

        cdef:
            double start_time = 0., end_time = 0.
            double build_time = 0., solve_time = 0.

            timespec ts

        self.only_dgels = 0.0
        self.first_point = False
        for point in points:
            if point == 52:
                self.first_point = True
            if grid.boundary_points[point] and not neumann_point[point]: 
                continue
            clock_gettime(CLOCK_REALTIME, &ts)
            start_time = ts.tv_sec + (ts.tv_nsec / 1e9)

            n_elem  = grid.esup_ptr[point + 1] - grid.esup_ptr[point]
            n_face  = grid.fsup_ptr[point + 1] - grid.fsup_ptr[point]
            n_bface = 0

            for i in range(grid.fsup_ptr[point], grid.fsup_ptr[point + 1]):
                face = grid.fsup[i]
                if grid.boundary_faces[face] == 1:
                    n_bface = n_bface + 1
            
            m    = 3 * n_elem + n_face + n_bface
            n    = 3 * n_elem + 1
            nrhs = n_elem + neumann_point[point]
            lda  = max(1, m)
            ldb  = max(1, m)

            for i in range(M_MAX):
                for j in range(N_MAX):
                    Mi[i, j] = 0.0

            for i in range(M_MAX):
                for j in range(NRHS_MAX):
                    Ni[i, j] = 0.0

            self.build_ks_sv_arrays(grid, point, 
                                    KSetv, Sv, Svb, 
                                    n_elem, n_face, n_bface)
            
            self.build_ls_matrices(grid, point, KSetv, Sv, Svb, 
                                   n_elem, n_face, n_bface, 
                                   permeability, diff_mag, 
                                   xv, xK, dKv, 
                                   xS, N_sj, Ks_Sv, eta_j, 
                                   T_sj1, T_sj2, tau_j2, tau_tsj2, 
                                   nL1, nL2, Ij1, Ij2, temp_cross,
                                   idx1, idx2, idx3,
                                   Mi, Ni)

            if neumann_point[point]:
                self.set_neumann_rows(grid, point, KSetv, Sv, Svb, 
                                      n_elem, n_face, n_bface, 
                                      permeability, neumann_val, 
                                      neumann_rows, Ks_Svb, 
                                      nL, Ik,
                                      Mi, Ni)

            clock_gettime(CLOCK_REALTIME, &ts)
            end_time = ts.tv_sec + (ts.tv_nsec / 1e9)
            build_time += end_time - start_time

            clock_gettime(CLOCK_REALTIME, &ts)
            
            start_time = ts.tv_sec + (ts.tv_nsec / 1e9)
            
            self.solve_ls(point, neumann_point[point], 
                          Mi, Ni, 
                          m, n, nrhs,
                          lda, ldb,
                          work, lwork,
                          weights, neumann_ws)
            
            if self.logging and self.first_point:
                self.logger.json("first_point", self.log_dict)

            clock_gettime(CLOCK_REALTIME, &ts)
            end_time = ts.tv_sec + (ts.tv_nsec / 1e9)
            solve_time += end_time - start_time

            
            
            self.first_point = False
            
        
        if self.logging:
            self.logger.log(f"GLS: build {build_time:.2f} s, Solve {solve_time:.2f} s, DGELS {self.only_dgels:.2f}", "INFO")

    cdef view.array array(self, tuple shape, str t):
        if t == 'i':
            return view.array(shape=shape, itemsize=sizeof(int), format="i", mode='c')
        elif t == 'l':
            return view.array(shape=shape, itemsize=sizeof(long), format="l", mode='c')
        elif t == 'd':
            return view.array(shape=shape, itemsize=sizeof(double), format="d", mode='c')
        else:
            raise ValueError("Invalid type")

    cdef void build_ks_sv_arrays(self, Grid grid, int point, 
                                 DTYPE_I_t[::1] KSetv, DTYPE_I_t[::1] Sv, DTYPE_I_t[::1] Svb, 
                                 const int n_elem, const int n_face, const int n_bface):
        cdef:
            int i, j
            int face
    
        for i in range(grid.esup_ptr[point], grid.esup_ptr[point + 1]):
            KSetv[i - grid.esup_ptr[point]] = grid.esup[i]
        j = 0
        for i in range(grid.fsup_ptr[point], grid.fsup_ptr[point + 1]):
            face = grid.fsup[i]
            Sv[i - grid.fsup_ptr[point]] = face
            if grid.boundary_faces[face] == 1:
                Svb[j] = face
                j = j + 1
        
        cdef:
            dict temp_dict
        
        if self.logging and self.first_point:
            temp_dict = {
                "point": point,
                "KSetv": KSetv[:n_elem],
                "Sv": Sv[:n_face],
                "Svb": Svb[:n_bface]
            }
            for key in temp_dict:
                try:
                    self.log_dict[key] = np.asarray(temp_dict[key]) 
                except:
                    self.log_dict[key] = temp_dict[key]


    cdef void build_ls_matrices(self, Grid grid, int point, 
                                const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb, 
                                const int n_elem, const int n_face, const int n_bface, 
                                DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] diff_mag,
                                DTYPE_F_t[::1] xv, DTYPE_F_t[:, ::1] xK, DTYPE_F_t[:, ::1] dKv,
                                DTYPE_F_t[:, ::1] xS, DTYPE_F_t[:, ::1] N_sj, DTYPE_I_t[:, ::1] Ks_Sv, DTYPE_F_t[::1] eta_j,
                                DTYPE_F_t[:, ::1] T_sj1, DTYPE_F_t[:, ::1] T_sj2, DTYPE_F_t[::1] tau_j2, DTYPE_F_t[:, ::1] tau_tsj2,
                                DTYPE_F_t[:, ::1] nL1, DTYPE_F_t[:, ::1] nL2, DTYPE_I_t[::1] Ij1, DTYPE_I_t[::1] Ij2, DTYPE_F_t[::1] temp_cross,
                                DTYPE_I_t[::1] idx1, DTYPE_I_t[::1] idx2, DTYPE_I_t[::1] idx3,
                                DTYPE_F_t[::1, :] Mi, DTYPE_F_t[::1, :] Ni):

        cdef:
            int i, j, k

        if n_bface >= n_face:
            return
        xv = grid.point_coords[point]
        for i in range(n_elem):
            xK[i] = grid.centroids[KSetv[i]]
            dKv[i, 0] = xK[i, 0] - xv[0]
            dKv[i, 1] = xK[i, 1] - xv[1]
            dKv[i, 2] = xK[i, 2] - xv[2]

        for i in range(n_elem):
            
            Mi[i, 3 * i]     = dKv[i, 0]
            Mi[i, 3 * i + 1] = dKv[i, 1]
            Mi[i, 3 * i + 2] = dKv[i, 2]
            Mi[i, 3 * n_elem] = 1.0
            Ni[i, i] = 1.0
        cdef:
            int n_esuf
            int n_iface = n_face - n_bface
        j = 0
        cdef:
            int n_dot = 3, m_dot = 3
            int incx = 1,  incy = 1
            DTYPE_F_t alpha = 1.0, beta = 0.0

        for i in range(n_face):
            
            n_esuf = grid.esuf_ptr[Sv[i] + 1] - grid.esuf_ptr[Sv[i]]

            if n_esuf < 2:
                continue

            xS[j]   = grid.faces_centers[Sv[i]]
            N_sj[j] = grid.normal_faces[Sv[i]]

            eta_j[j] = 0.0
            for k in range(n_esuf):
                Ks_Sv[j, k] = grid.esuf[grid.esuf_ptr[Sv[i]] + k]
                eta_j[j]    = max(eta_j[j], diff_mag[Ks_Sv[j, k]])
            
            T_sj1[j, 0] = xv[0] - xS[j, 0]
            T_sj1[j, 1] = xv[1] - xS[j, 1]
            T_sj1[j, 2] = xv[2] - xS[j, 2]
            
            self.cross(N_sj[j], T_sj1[j], temp_cross)  
            T_sj2[j, 0] = temp_cross[0]
            T_sj2[j, 1] = temp_cross[1]
            T_sj2[j, 2] = temp_cross[2]
            tau_j2[j]   = self.norm(T_sj2[j]) ** (-eta_j[j])

            tau_tsj2[j, 0] = tau_j2[j] * T_sj2[j, 0]
            tau_tsj2[j, 1] = tau_j2[j] * T_sj2[j, 1]
            tau_tsj2[j, 2] = tau_j2[j] * T_sj2[j, 2]

            blas.dgemv("T", &m_dot, &n_dot, &alpha, &permeability[Ks_Sv[j, 0], 0, 0], &m_dot, &N_sj[j, 0], &incx, &beta, &nL1[j, 0], &incy)
            blas.dgemv("T", &m_dot, &n_dot, &alpha, &permeability[Ks_Sv[j, 1], 0, 0], &m_dot, &N_sj[j, 0], &incx, &beta, &nL2[j, 0], &incy)
            
            j += 1 

        cdef:
            unordered_map[DTYPE_I_t, DTYPE_I_t] KSetv_map
        
        for i in range(n_elem):
            KSetv_map[KSetv[i]] = i
        
        for i in range(n_iface):
            Ij1[i] = KSetv_map[Ks_Sv[i, 0]]
            Ij2[i] = KSetv_map[Ks_Sv[i, 1]]
        
        cdef:
            
            int start = n_elem
            int stop  = n_elem + 3 * n_face - 2
        
        for i in range(n_iface):
            idx1[i] = start
            idx2[i] = start + 1
            idx3[i] = start + 2
            start += 3

        

        for i in range(n_iface):
            self._set_mi(idx1[i], 3 * Ij1[i], nL1[i], Mi,-1)
            self._set_mi(idx1[i], 3 * Ij2[i], nL2[i], Mi, 1)

            self._set_mi(idx2[i], 3 * Ij1[i], T_sj1[i], Mi,-1)
            self._set_mi(idx2[i], 3 * Ij2[i], T_sj1[i], Mi, 1)

            self._set_mi(idx3[i], 3 * Ij1[i], tau_tsj2[i], Mi,-1)
            self._set_mi(idx3[i], 3 * Ij2[i], tau_tsj2[i], Mi, 1)
        
        cdef:
            str key
            dict temp_dict
        if self.logging and self.first_point:

            temp_dict = {
                "n_iface": n_iface,
                "xv": xv,
                "xK": xK[:n_elem],
                "dKv": dKv[:n_elem],
                "xS": xS[:n_iface],
                "N_sj": N_sj[:n_iface],
                "Ks_Sv": Ks_Sv[:n_iface],
                "eta_j": eta_j[:n_iface],
                "T_sj1": T_sj1[:n_iface],
                "T_sj2": T_sj2[:n_iface],
                "tau_j2": tau_j2[:n_iface],
                "tau_tsj2": tau_tsj2[:n_iface],
                "nL1": nL1[:n_iface],
                "nL2": nL2[:n_iface],
                "Ij1": Ij1[:n_iface],
                "Ij2": Ij2[:n_iface],
                "idx1": idx1[:n_iface],
                "idx2": idx2[:n_iface],
                "idx3": idx3[:n_iface],
                "Mi": Mi,
                "Ni": Ni
            }
            for key in temp_dict:
                try:
                    self.log_dict[key] = np.asarray(temp_dict[key]) 
                except:
                    self.log_dict[key] = temp_dict[key]
        
    cdef void _set_mi(self, 
                     const int row, const int col, 
                     const DTYPE_F_t[::1] v, DTYPE_F_t[::1, :] Mi, int k):
            Mi[row, col]     = v[0] * k
            Mi[row, col + 1] = v[1] * k
            Mi[row, col + 2] = v[2] * k

    cdef void cross(self, const DTYPE_F_t[::1] a, const DTYPE_F_t[::1] b, DTYPE_F_t[::1] c):
    
        c[0] = a[1] * b[2] - a[2] * b[1]
        c[1] = a[2] * b[0] - a[0] * b[2]
        c[2] = a[0] * b[1] - a[1] * b[0]
    
    cdef DTYPE_F_t norm(self, const DTYPE_F_t[::1] a):
        return sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

    cdef void set_neumann_rows(self, Grid grid,
                               int point, const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb, 
                               const int n_elem, const int n_face, const int n_bface, 
                               DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] neumann_val,
                               DTYPE_I_t[::1] neumann_rows, DTYPE_I_t[:, ::1] Ks_Svb, 
                               DTYPE_F_t[:, ::1] nL, DTYPE_I_t[::1] Ik,
                               DTYPE_F_t[::1, :] Mi, DTYPE_F_t[::1, :] Ni):
        cdef:
            int i, j, k

        cdef:
            int start = n_elem + 3 * n_face
        
        cdef:
            int bpoint
            int total_bpoints
            int n_dot = 3, m_dot = 3
            int incx = 1,  incy = 1
            DTYPE_F_t alpha = 1.0, beta = 0.0

        for i in range(n_bface):
            neumann_rows[i] = start + i
            Ks_Svb[i] = grid.esuf[grid.esuf_ptr[Svb[i]]]
            blas.dgemv("T", &m_dot, &n_dot, &alpha, &permeability[Ks_Svb[i, 0], 0, 0], &m_dot, &grid.normal_faces[Svb[i], 0], &incx, &beta, &nL[i, 0], &incy)
            total_bpoints = 0
            for bpoint in grid.inpofa[Svb[i]]:
                if bpoint == -1:
                    break
                total_bpoints += 1
                Ni[neumann_rows[i], n_elem] += neumann_val[bpoint]
            Ni[neumann_rows[i], n_elem] /= total_bpoints
        
        cdef:
            unordered_map[DTYPE_I_t, DTYPE_I_t] KSetv_map
        
        for i in range(n_elem):
            KSetv_map[KSetv[i]] = i
        
        for i in range(n_bface):
            Ik[i] = KSetv_map[Ks_Svb[i, 0]]
            Mi[neumann_rows[i], 3 * Ik[i]]     = -nL[i, 0]
            Mi[neumann_rows[i], 3 * Ik[i] + 1] = -nL[i, 1]
            Mi[neumann_rows[i], 3 * Ik[i] + 2] = -nL[i, 2]
    
        
    
    cdef void solve_ls(self, int point, int is_neumann,
                       DTYPE_F_t[::1, :] Mi, DTYPE_F_t[::1, :] Ni, 
                       int m, int n, int nrhs,
                       int lda, int ldb,
                       DTYPE_F_t[::1] work, int lwork,
                       DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        cdef:
            int i, j
            int info = 0
            DTYPE_F_t[::1, :] A = Mi.copy_fortran()
            DTYPE_F_t[::1, :] B = Ni.copy_fortran()
        cdef:
            double start_time = 0., end_time = 0.
            timespec ts
        
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + (ts.tv_nsec / 1e9)
   
        lapack.dgels('N', &m, &n, &nrhs, &A[0, 0], &lda, &B[0, 0], &ldb, &work[0], &lwork, &info)
    
        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + (ts.tv_nsec / 1e9)

        if info:
            if self.logging:
                self.logger.log(f"Failed to solve LS system in point {point}. Info: {info}", "ERROR")
        
        self.only_dgels += end_time - start_time
        cdef:
            int M_size  = n
            int w_total = nrhs - is_neumann
        
        for i in range(w_total):
            weights[point, i] = B[M_size - 1, i]
        
        if is_neumann:
            neumann_ws[point] = B[M_size - 1, w_total - 1]

        cdef:
            str key
            dict temp_dict
        
        if self.logging and self.first_point:
            temp_dict = {
                "m": m,
                "n": n,
                "nrhs": nrhs,
                "lda": lda,
                "ldb": ldb,
                "lwork": lwork,
                "M_size": M_size,
                "w_total": w_total,
                "A": A,
                "B": B,
            }
            for key in temp_dict:
                try:
                    self.log_dict[key] = np.asarray(temp_dict[key]) 
                except:
                    self.log_dict[key] = temp_dict[key]
        
