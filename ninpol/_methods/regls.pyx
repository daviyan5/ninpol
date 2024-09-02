import numpy as np
import time
import re

from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
from libc.math cimport sqrt

from cython cimport view

DTYPE_I = int
DTYPE_F = float

cdef class GLSInterpolation:
    def __cinit__(self, int logging=False):
        self.logging  = logging
        self.log_dict = {}
        self.logger   = Logger("GLS")

    cdef void GLS(self, Grid grid, const DTYPE_I_t[::1] points, 
                  const DTYPE_F_t[:, :, ::1] permeability, 
                  const DTYPE_F_t[::1] diff_mag, 
                  const DTYPE_I_t[::1] neumann_point, const DTYPE_F_t[::1] neumann_val,
                  DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
    
        cdef:
            int point
            int n_points = points.shape[0]
        
        cdef:
            DTYPE_I_t[::1] KSetv
            DTYPE_I_t[::1] Sv
            DTYPE_I_t[::1] Svb
        
        cdef:
            DTYPE_I_t[:, ::1] Mi
            DTYPE_I_t[:, ::1] Ni

        if self.logging:
            self.logger.log(f"Starting GLS interpolation over {len(points)} points", "INFO")

        for point in points:
            self.build_ks_sv_arrays(grid, point, KSetv, Sv, Svb)
            self.build_ls_matrices(grid, point, KSetv, Sv, Svb, permeability, diff_mag, Mi, Ni)
            if neumann_point[point]:
                self.set_neumann_rows(grid, point, KSetv, Sv, Svb, permeability, neumann_val, Mi, Ni)
            self.solve_ls(Mi, Ni, weights, neumann_ws)
            if self.logging:
                self.log_dict[point] = {
                    "point": point,
                    "KSetv": KSetv,
                    "Sv":    Sv,
                    "Svb":   Svb,
                    "Mi":    Mi,
                    "Ni":    Ni
                }
        if self.logging:
            # Convert every memoryview to a list
            def convert_to_arr(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        convert_to_arr(v)
                    try:
                        d[k] = np.asarray(v)
                    except:
                        pass
            self.logger.json("Points", self.log_dict)

    cdef void array(self, const tuple shape, const str t):
        if t == 'i':
            return view.array(shape=shape, itemsize=sizeof(DTYPE_I_t), format="i", mode='c')
        elif t == 'd':
            return view.array(shape=shape, itemsize=sizeof(DTYPE_F_t), format="d", mode='c')
        else:
            raise ValueError("Invalid type")

    cdef void build_ks_sv_arrays(self, const Grid grid, int point, 
                                 DTYPE_I_t[::1] KSetv, DTYPE_I_t[::1] Sv, DTYPE_I_t[::1] Svb):
        cdef:
            int i, j
            int n_elem = self.grid_obj.esup_ptr[point + 1] - self.grid_obj.esup_ptr[point]
            int n_face = self.grid_obj.fsup_ptr[point + 1] - self.grid_obj.fsup_ptr[point]
            int n_bface = 0
            int face
        
        
        for i in range(grid.fsup_ptr[point], grid.fsup_ptr[point + 1]):
            face = grid.fsup[i]
            if grid.boundary_faces[face] == 1:
                n_bface += 1

        KSetv = self.array((n_elem,),  "i")
        Sv    = self.array((n_face,),  "i")
        Svb   = self.array((n_bface,), "i")

        for i in range(grid.esup_ptr[point], grid.esup_ptr[point + 1]):
            KSetv[i - grid.esup_ptr[point]] = grid.esup[i]
        
        j = 0
        for i in range(grid.fsup_ptr[point], grid.fsup_ptr[point + 1]):
            face = grid.fsup[i]
            Sv[i - grid.fsup_ptr[point]] = face
            if grid.boundary_faces[face] == 1:
                Svb[j] = face
                j += 1

    cdef void build_ls_matrices(self, const Grid grid, int point, 
                                const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb,
                                const DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] diff_mag,
                                DTYPE_I_t[:, ::1] Mi, DTYPE_I_t[:, ::1] Ni):

        cdef:
            int n_elems  = KSetv.shape[0]
            int n_faces  = Sv.shape[0]
            int n_bfaces = Svb.shape[0]
            int i, j, k

        cdef:
            DTYPE_F_t[::1] xv  = grid.point_coords[point]
            DTYPE_F_t[:, ::1] xK  = self.array((n_elems, 3), "d")
            DTYPE_F_t[:, ::1] dKv = self.array((n_elems, 3), "d")

        for i in range(n_elems):
            xK[i] = grid.centroids[KSetv[i]]
            dKv[i] = xK[i] - xv
        
        for i in range(n_elems):
            for j in range(0, 3 * n_elems, 3):
                Mi[i, j]     = dKv[i, 0]
                Mi[i, j + 1] = dKv[i, 1]
                Mi[i, j + 2] = dKv[i, 2]
            Mi[i, 3 * n_elems] = 1.0
            Ni[i, i] = 1.0
        
        cdef:
            int n_esuf
            DTYPE_F_t[:, ::1] xS    = self.array((n_faces - n_bfaces, 3), "d")
            DTYPE_F_t[:, ::1] N_sj  = self.array((n_faces - n_bfaces, 3), "d")
            DTYPE_I_t[:, ::1] Ks_Sv = self.array((n_faces - n_bfaces, NinpolSizes.NINPOL_MAX_ELEMENTS_PER_FACE), "i")
            
            
            DTYPE_F_t[::1] eta_j    = self.array((n_faces - n_bfaces,), "d")
            DTYPE_F_t[:, ::1] T_sj1 = self.array((n_faces - n_bfaces, 3), "d")
            DTYPE_F_t[:, ::1] T_sj2 = self.array((n_faces - n_bfaces, 3), "d")

            DTYPE_F_t[::1] tau_j2      = self.array((n_faces - n_bfaces,), "d")
            DTYPE_F_t[:, ::1] tau_tsj2 = self.array((n_faces - n_bfaces, 3), "d")

            DTYPE_F_t[::1] nL1 = self.array((n_faces - n_bfaces,), "d")
            DTYPE_F_t[::1] nL2 = self.array((n_faces - n_bfaces,), "d")
        
        j = 0
        for i in range(n_faces):
            
            n_esuf = grid.esuf_ptr[Sv[i] + 1] - grid.esuf_ptr[Sv[i]]

            if n_esuf < 2:
                continue

            xS[j]   = grid.face_centers[Sv[i]]
            N_sj[j] = grid.normal_faces[Sv[i]]

            for k in range(n_esuf):
                Ks_Sv[j, k] = grid.esuf[grid.esuf_ptr[Sv[i]] + k]
                eta_j[j]    = max(eta_j[j], diff_mag[Ks_Sv[j, k]])
            
            T_sj1[j, 0] = xv[0] - xS[j, 0]
            T_sj1[j, 1] = xv[1] - xS[j, 1]
            T_sj1[j, 2] = xv[2] - xS[j, 2]
            
            T_sj2[j]    = self.cross(N_sj[j], T_sj1[j])  
            tau_j2[j]   = blas.nrm2(&T_sj2[j]) ** (-eta_j[j])

            tau_tsj2[j, 0] = tau_j2[j] * T_sj2[j, 0]
            tau_tsj2[j, 1] = tau_j2[j] * T_sj2[j, 1]
            tau_tsj2[j, 2] = tau_j2[j] * T_sj2[j, 2]

            nL1[j] = blas.ddot(&N_sj[j], permeability[Ks_Sv[j, 0]])
            nL2[j] = blas.ddot(&N_sj[j], permeability[Ks_Sv[j, 1]])
            j += 1 

        cdef:
            DTYPE_I_t[::1] Ij1 = self.array((n_faces - n_bfaces), "i")  # Ij1[i] = index of Ks_Sv[i, 0] in KSetv
            DTYPE_I_t[::1] Ij2 = self.array((n_faces - n_bfaces), "i")  # Ij2[i] = index of Ks_Sv[i, 1] in KSetv

        cdef:
            unordered_map[DTYPE_I_t, DTYPE_I_t] KSetv_map
            dict temp_dict = {}
        
        for i in range(n_elems):
            KSetv_map[KSetv[i]] = i
        
        for i in range(n_faces - n_bfaces):
            Ij1[i] = KSetv_map[Ks_Sv[i, 0]]
            Ij2[i] = KSetv_map[Ks_Sv[i, 1]]
        
        cdef:
            DTYPE_I_t[::1] idx1 = self.array((n_faces - n_bfaces), "i")
            DTYPE_I_t[::1] idx2 = self.array((n_faces - n_bfaces), "i")
            DTYPE_I_t[::1] idx3 = self.array((n_faces - n_bfaces), "i")
            int start = n_elems
            int stop  = n_elems + 3 * n_faces - 2
        
        for i in range(n_faces - n_bfaces):
            idx1[i] = start
            idx2[i] = start + 1
            idx3[i] = start + 2
            start += 3

        cdef void set_mi(int row, int col, DTYPE_F_t[::1] v, int k = 1):
            Mi[row, col]     = v[0] * k
            Mi[row, col + 1] = v[1] * k
            Mi[row, col + 2] = v[2] * k

        for i in range(n_faces - n_bfaces):
            set_mi(idx1[i], 3 * Ij1[i], nL1[i], -1)
            set_mi(idx1[i], 3 * Ij2[i], nL2[i])

            set_mi(idx2[i], 3 * Ij1[i], T_sj1[i], -1)
            set_mi(idx2[i], 3 * Ij2[i], T_sj1[i])

            set_mi(idx3[i], 3 * Ij1[i], tau_tsj2[i], -1)
            set_mi(idx3[i], 3 * Ij2[i], tau_tsj2[i])
        
        if self.logging:
            temp_dict = {
                "xv": xv,
                "xK": xK,
                "dKv": dKv,

                "xS": xS,
                "N_sj": N_sj,

                "eta_j": eta_j,
                "Ks_Sv": Ks_Sv,

                "Ij1": Ij1,
                "Ij2": Ij2,

                "T_sj1": T_sj1,
                "T_sj2": T_sj2,

                "tau_j2": tau_j2,
                "tau_tsj2": tau_tsj2,

                "nL1": nL1,
                "nL2": nL2,
                
                "idx1": idx1,
                "idx2": idx2,
                "idx3": idx3
            }
            for k, v in temp_dict.items():
                self.log_dict[point][k] = v
        

    cdef void cross(self, const DTYPE_F_t[::1] a, const DTYPE_F_t[::1] b):
        cdef:
            DTYPE_F_t[::1] c = self.array((3,), "d")
        
        c[0] = a[1] * b[2] - a[2] * b[1]
        c[1] = a[2] * b[0] - a[0] * b[2]
        c[2] = a[0] * b[1] - a[1] * b[0]
        
        return c

    cdef void set_neumann_rows(self, const Grid grid,
                               int point, const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_I_t[::1] Svb,
                               const DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] neumann_val,
                               DTYPE_I_t[:, ::1] Mi, DTYPE_I_t[:, ::1] Ni):
        cdef:
            int n_elems  = KSetv.shape[0]
            int n_faces  = Sv.shape[0]
            int n_bfaces = Svb.shape[0]
            int i, j, k

        cdef:
            int start = n_elems + 3 * n_faces
            DTYPE_I_t[::1] neumann_rows = self.array(n_bfaces, "i")
            DTYPE_I_t[:, ::1] Ks_Svb = self.array((n_bfaces, NinpolSizes.NINPOL_MAX_ELEMENTS_PER_FACE), "i")
            DTYPE_F_t[:, ::1] nL = self.array((n_bfaces, 3), "d")
        
        for i in range(n_bfaces):
            neumann_rows[i] = start + i
            Ks_Svb[j] = grid.esuf[grid.esuf_ptr[Svb[i]]]
            nL[j] = blas.ddot(&grid.normal_faces[Svb[i]], permeability[Ks_Svb[j]])
            Ni[neumann_rows[i], n_elems] = neumann_val[Svb[i]]
        
        cdef:
            unordered_map[DTYPE_I_t, DTYPE_I_t] KSetv_map
        
        for i in range(n_elems):
            KSetv_map[KSetv[i]] = i
        
        cdef:
            DTYPE_I_t[::1] Ik = self.array(n_bfaces, "i")
        
        for i in range(n_bfaces):
            Ik[i] = KSetv_map[Ks_Svb[i, 0]]
            Mi[neumann_rows[i], 3 * Ik[i]]     = -nL[i, 0]
            Mi[neumann_rows[i], 3 * Ik[i] + 1] = -nL[i, 1]
            Mi[neumann_rows[i], 3 * Ik[i] + 2] = -nL[i, 2]
        
    
    cdef void solve_ls(self, int point, int is_neumann,
                       DTYPE_I_t[:, ::1] Mi, DTYPE_I_t[:, ::1] Ni, 
                       DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
        cdef:
            int m = Mi.shape[0]
            int n = Mi.shape[1]
            int nrhs = Ni.shape[1]
            const int lda = max(1, m)
            const int ldb = max(1, m)

            DTYPE_I_t[::1] jptv = self.array((n,), "i")
            float cond = 1e-12
            DTYPE_I_t[::1] work = self.array((1,), "i")
            int lwork = -1
            int info = 0
        
        lapack.dgelsy(&m, &n, &nrhs, &Mi, &lda, &Ni, &ldb, &jptv, &cond, &lwork, &info)
        
        lwork = work[0]
        work = self.array((lwork,), "i")
        lapack.dgelsy(&m, &n, &nrhs, &Mi, &lda, &Ni, &ldb, &jptv, &cond, &lwork, &info)

        if info:
            self.logger.log(f"Failed to solve LS system. Info: {info}", "ERROR")
            raise ValueError("Failed to solve LS system")

        cdef:
            int i, j
            int M_size  = Ni.shape[0]
            int w_total = Ni.shape[1] - is_neumann
        
        for i in range(w_total):
            weights[point, i] = Ni[M_size - 1, i]
        
        if is_neumann:
            neumann_ws[point] = Ni[M_size - 1, w_total - 1]
        
        
