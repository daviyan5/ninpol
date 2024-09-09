import numpy as np

from libcpp.unordered_map cimport unordered_map
from libc.math cimport sqrt

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

from cython cimport view

DTYPE_I = int
DTYPE_F = float

cdef class GLSInterpolation:
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
        
        cdef:
            int dim = grid.dim
            int permeability_index = variable_to_index["cells"]["permeability"]
            int diff_mag_index     = variable_to_index["cells"]["diff_mag"]
            int neumann_flag_index = variable_to_index["points"]["neumann_flag" + "_" + variable]
            int neumann_val_index  = variable_to_index["points"]["neumann" + "_" + variable]

            DTYPE_F_t[:, :, ::1] permeability   = np.reshape(cells_data[permeability_index], 
                                                            (grid.n_elems, dim, dim))
            
            const DTYPE_F_t[::1] diff_mag             = cells_data[diff_mag_index]

            const DTYPE_I_t[::1] neumann_point        = np.asarray(points_data[neumann_flag_index]).astype(DTYPE_I)

            const DTYPE_F_t[::1] neumann_val          = points_data[neumann_val_index]
                
        self.GLS(grid, target_points, permeability, diff_mag, neumann_point, neumann_val, weights, neumann_ws)
        
    cdef void GLS(self, Grid grid, const DTYPE_I_t[::1] points, 
                  DTYPE_F_t[:, :, ::1] permeability, 
                  const DTYPE_F_t[::1] diff_mag, 
                  const DTYPE_I_t[::1] neumann_point, const DTYPE_F_t[::1] neumann_val,
                  DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
    
        cdef:
            int point
            int n_points = points.shape[0]
        
        cdef:
            DTYPE_I_t[::1] KSetv = self.array((grid.MX_ELEMENTS_PER_POINT,), "l")
            DTYPE_I_t[::1] Sv  = self.array((grid.MX_FACES_PER_POINT,), "l")
            DTYPE_I_t[::1] Svb = self.array((max(1, grid.MX_FACES_PER_POINT),), "l")
        
        cdef:
            DTYPE_F_t[::1, :] Mi = view.array((grid.MX_ELEMENTS_PER_POINT + 3 * grid.MX_FACES_PER_POINT + grid.MX_FACES_PER_POINT, 3 * grid.MX_ELEMENTS_PER_POINT + 1), "d", mode='fortran')
            DTYPE_F_t[::1, :] Ni = view.array((grid.MX_ELEMENTS_PER_POINT + 3 * grid.MX_FACES_PER_POINT + grid.MX_FACES_PER_POINT, grid.MX_ELEMENTS_PER_POINT + 1), "d", mode='fortran')

        cdef:
            DTYPE_F_t[::1] xv = self.array((3,), "d")
            DTYPE_F_t[:, ::1] xK  = self.array((grid.MX_ELEMENTS_PER_POINT, 3), "d")
            DTYPE_F_t[:, ::1] dKv = self.array((grid.MX_ELEMENTS_PER_POINT, 3), "d")

        cdef:
            DTYPE_F_t[:, ::1] xS    = self.array((grid.MX_FACES_PER_POINT, 3), "d")
            DTYPE_F_t[:, ::1] N_sj  = self.array((grid.MX_FACES_PER_POINT, 3), "d")
            DTYPE_I_t[:, ::1] Ks_Sv = self.array((grid.MX_FACES_PER_POINT, NinpolSizes.NINPOL_MAX_ELEMENTS_PER_FACE), "l")
            
            
            DTYPE_F_t[::1] eta_j    = self.array((grid.MX_FACES_PER_POINT,), "d")
            DTYPE_F_t[:, ::1] T_sj1 = self.array((grid.MX_FACES_PER_POINT, 3), "d")
            DTYPE_F_t[:, ::1] T_sj2 = self.array((grid.MX_FACES_PER_POINT, 3), "d")

            DTYPE_F_t[::1] tau_j2      = self.array((grid.MX_FACES_PER_POINT,), "d")
            DTYPE_F_t[:, ::1] tau_tsj2 = self.array((grid.MX_FACES_PER_POINT, 3), "d")

            DTYPE_F_t[:, ::1] nL1 = self.array((grid.MX_FACES_PER_POINT, 3), "d")
            DTYPE_F_t[:, ::1] nL2 = self.array((grid.MX_FACES_PER_POINT, 3), "d")

            DTYPE_I_t[::1] Ij1 = self.array((grid.MX_FACES_PER_POINT,), "l")  # Ij1[i] = index of Ks_Sv[i, 0] in KSetv
            DTYPE_I_t[::1] Ij2 = self.array((grid.MX_FACES_PER_POINT,), "l")  # Ij2[i] = index of Ks_Sv[i, 1] in KSetv
            
            DTYPE_I_t[::1] idx1 = self.array((grid.MX_FACES_PER_POINT,), "l")
            DTYPE_I_t[::1] idx2 = self.array((grid.MX_FACES_PER_POINT,), "l")
            DTYPE_I_t[::1] idx3 = self.array((grid.MX_FACES_PER_POINT,), "l")

            DTYPE_I_t[::1] neumann_rows = self.array((grid.MX_FACES_PER_POINT,), "l")
            DTYPE_I_t[:, ::1] Ks_Svb = self.array((grid.MX_FACES_PER_POINT, NinpolSizes.NINPOL_MAX_ELEMENTS_PER_FACE), "l")
            DTYPE_F_t[:, ::1] nL = self.array((grid.MX_FACES_PER_POINT, 3), "d")

            DTYPE_I_t[::1] Ik = self.array((grid.MX_FACES_PER_POINT,), "l")

            DTYPE_F_t[::1] temp = self.array((3,), "d")

            int m = Mi.shape[0]
            int n = Mi.shape[1]
            int nrhs = Ni.shape[1]
            int lda = max(1, m)
            int ldb = max(1, m)
            int lwork = -1
            int info = 0

            double[::1] work = self.array((1,), "d")

        lapack.dgels('N', &m, &n, &nrhs, &Mi[0, 0], &lda, &Ni[0, 0], &ldb, &work[0], &lwork, &info)
        
        lwork = int(work[0])
        work = self.array((lwork,), "d")

        cdef:
            int n_elem
            int n_face
            int face
            int n_bface
        
        cdef:
            double start_time = 0., end_time = 0.
            double build_time = 0., solve_time = 0.

            timespec ts


        for point in points:
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
                    n_bface += 1
            
            for j in range(n):
                for i in range(m):
                    Mi[i, j] = 0.0

            for j in range(nrhs):
                for i in range(m):    
                    Ni[i, j] = 0.0

            self.build_ks_sv_arrays(grid, point, n_elem, n_face, KSetv, Sv, Svb, n_bface)
            self.build_ls_matrices(grid, point, KSetv, Sv, Svb, n_elem, n_face, n_bface, permeability, diff_mag, 
                                   Mi, Ni, temp, xK, dKv, xv, xS, N_sj, Ks_Sv, eta_j, T_sj1, T_sj2, tau_j2, tau_tsj2, nL1, nL2, Ij1, Ij2, idx1, idx2, idx3)
            if neumann_point[point]:
                self.set_neumann_rows(grid, point, KSetv, Sv, Svb, n_bface, n_elem, n_face, permeability, neumann_val, Mi, Ni, Ks_Svb, nL, neumann_rows, Ik)

            clock_gettime(CLOCK_REALTIME, &ts)
            end_time = ts.tv_sec + (ts.tv_nsec / 1e9)
            build_time += end_time - start_time

            clock_gettime(CLOCK_REALTIME, &ts)
            start_time = ts.tv_sec + (ts.tv_nsec / 1e9)

            self.solve_ls(point, neumann_point[point], m, n, nrhs, lda, ldb, lwork, info, Mi, Ni, work, weights, neumann_ws)

            clock_gettime(CLOCK_REALTIME, &ts)
            end_time = ts.tv_sec + (ts.tv_nsec / 1e9)
            solve_time += end_time - start_time
        
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
                                 const int n_elem, const int n_face,
                                 DTYPE_I_t[::1] KSetv, DTYPE_I_t[::1] Sv, DTYPE_I_t[::1] Svb, 
                                 const int n_bface) nogil:
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
                j += 1

    cdef void build_ls_matrices(self, Grid grid, int point, 
                                const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb, 
                                const int n_elem, const int n_face, const int n_bface, 
                                DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] diff_mag,
                                DTYPE_F_t[::1, :] Mi, DTYPE_F_t[::1, :] Ni,
                                DTYPE_F_t[::1] temp, DTYPE_F_t[:, ::1] xK, DTYPE_F_t[:, ::1] dKv, DTYPE_F_t[::1] xv,
                                DTYPE_F_t[:, ::1] xS, DTYPE_F_t[:, ::1] N_sj, DTYPE_I_t[:, ::1] Ks_Sv,
                                DTYPE_F_t[::1] eta_j, DTYPE_F_t[:, ::1] T_sj1, DTYPE_F_t[:, ::1] T_sj2,
                                DTYPE_F_t[::1] tau_j2, DTYPE_F_t[:, ::1] tau_tsj2, DTYPE_F_t[:, ::1] nL1, DTYPE_F_t[:, ::1] nL2,
                                DTYPE_I_t[::1] Ij1, DTYPE_I_t[::1] Ij2, DTYPE_I_t[::1] idx1, DTYPE_I_t[::1] idx2, DTYPE_I_t[::1] idx3) nogil:

        cdef:
            int i, j, k

        if n_bface >= n_face:
            return
            
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
            
            self.cross(N_sj[j], T_sj1[j], temp)
            T_sj2[j]    = temp  
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
        
        for i in range(n_face - n_bface):
            Ij1[i] = KSetv_map[Ks_Sv[i, 0]]
            Ij2[i] = KSetv_map[Ks_Sv[i, 1]]
        
        cdef:
            int start = n_elem
            int stop  = n_elem + 3 * n_face - 2
        
        for i in range(n_face - n_bface):
            idx1[i] = start
            idx2[i] = start + 1
            idx3[i] = start + 2
            start += 3

        

        for i in range(n_face - n_bface):
            self._set_mi(idx1[i], 3 * Ij1[i], nL1[i], Mi,-1)
            self._set_mi(idx1[i], 3 * Ij2[i], nL2[i], Mi, 1)

            self._set_mi(idx2[i], 3 * Ij1[i], T_sj1[i], Mi,-1)
            self._set_mi(idx2[i], 3 * Ij2[i], T_sj1[i], Mi, 1)

            self._set_mi(idx3[i], 3 * Ij1[i], tau_tsj2[i], Mi,-1)
            self._set_mi(idx3[i], 3 * Ij2[i], tau_tsj2[i], Mi, 1)
        
        
        
    cdef void _set_mi(self, 
                     const int row, const int col, 
                     const DTYPE_F_t[::1] v, DTYPE_F_t[::1, :] Mi, int k) noexcept nogil:
            Mi[row, col]     = v[0] * k
            Mi[row, col + 1] = v[1] * k
            Mi[row, col + 2] = v[2] * k

    cdef void cross(self, const DTYPE_F_t[::1] a, const DTYPE_F_t[::1] b, DTYPE_F_t[::1] c) noexcept nogil:
        c[0] = a[1] * b[2] - a[2] * b[1]
        c[1] = a[2] * b[0] - a[0] * b[2]
        c[2] = a[0] * b[1] - a[1] * b[0]
    
    cdef DTYPE_F_t norm(self, const DTYPE_F_t[::1] a) noexcept nogil:
        return sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

    cdef void set_neumann_rows(self, Grid grid,
                               int point, const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb, 
                               const int n_bface, const int n_elem, const int n_face,
                               DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] neumann_val,
                               DTYPE_F_t[::1, :] Mi, DTYPE_F_t[::1, :] Ni,
                               DTYPE_I_t[:, ::1] Ks_Svb, DTYPE_F_t[:, ::1] nL, DTYPE_I_t[::1] neumann_rows, DTYPE_I_t[::1] Ik) nogil:
        cdef:
            int i, j, k

        cdef:
            int start = n_elem + 3 * n_face
        
        cdef:
            int n_dot = 3, m_dot = 3
            int incx = 1,  incy = 1
            DTYPE_F_t alpha = 1.0, beta = 0.0

        for i in range(n_bface):
            neumann_rows[i] = start + i
            Ks_Svb[i] = grid.esuf[grid.esuf_ptr[Svb[i]]]
            blas.dgemv("T", &m_dot, &n_dot, &alpha, &permeability[Ks_Svb[i, 0], 0, 0], &m_dot, &grid.normal_faces[Svb[i], 0], &incx, &beta, &nL[i, 0], &incy)
            Ni[neumann_rows[i], n_elem] = neumann_val[Svb[i]]
        
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
                       int m, int n, int nrhs, int lda, int ldb, int lwork, int info,
                       DTYPE_F_t[::1, :] Mi, DTYPE_F_t[::1, :] Ni,
                       DTYPE_F_t[::1] work,
                       DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws) nogil:
        cdef:
            int i, j
        
        lapack.dgels('N', &m, &n, &nrhs, &Mi[0, 0], &lda, &Ni[0, 0], &ldb, &work[0], &lwork, &info)

        cdef:
            int M_size  = n
            int w_total = nrhs - is_neumann
        
        for i in range(w_total):
            weights[point, i] = Ni[M_size - 1, i]
        
        if is_neumann:
            neumann_ws[point] = Ni[M_size - 1, w_total - 1]
        
        
