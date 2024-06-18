import numpy as np

from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from libc.stdio cimport printf
from libc.math cimport sqrt

DTYPE_I = np.int64
DTYPE_F = np.float64

cdef void GLS(  Grid grid, 
                const DTYPE_I_t[::1] in_points, const DTYPE_I_t[::1] nm_points, 
                const DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] diff_mag,
                DTYPE_I_t[:, ::1] connectivity_idx,
                DTYPE_F_t[:, ::1] weights, DTYPE_F_t[::1] neumann_ws):
    """
    Main GLS function that processes the grid structure into substructures.
    """
    cdef:
        int n_vols = grid.n_elems
        int n_faces = grid.n_faces
        

cdef void interpolate_nodes(Grid grid, 
                            const DTYPE_F_t[::1] target,
                            DTYPE_F_t[:, ::1] weights,
                            DTYPE_F_t[::1] neumann_ws,
                            int is_neumann = 0):
    
    cdef:
        int i, j, k
        int n_target = target.shape[0]
        int el_idx, fc_idx
        int nK, nS
        DTYPE_F_t[:, ::1] normal_faces = grid.normal_faces

        DTYPE_F_t[::1] KSetv = np.zeros(grid.MX_ELEMENTS_PER_POINT, dtype=DTYPE_F)
        DTYPE_F_t[::1] Sv    = np.zeros(grid.MX_FACES_PER_POINT, dtype=DTYPE_F)

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] Mi = np.zeros((1, 1), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] Ni = np.zeros((1, 1), dtype=DTYPE_F)
        
        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] neu_rows = np.zeros(1, dtype=DTYPE_I)
        
        
        DTYPE_F_t[:, ::1] M = np.zeros((1, 1), dtype=DTYPE_F)

    for i in range(n_target):
        el_idx = 0
        for j in range(grid.esup_ptr[i], grid.esup_ptr[i + 1]):
            KSetv[el_idx] = grid.esup[j]
            el_idx = el_idx + 1

        nK = el_idx
        for j in range(el_idx, grid.MX_ELEMENTS_PER_POINT):
            KSetv[j] = -1

        fc_idx = 0
        for j in range(grid.fsup_ptr[i], grid.fsup_ptr[i + 1]):
            Sv[fc_idx] = grid.fsup[j]
            fc_idx = fc_idx + 1
        nS = fc_idx
        for j in range(fc_idx, grid.MX_FACES_PER_POINT):
            Sv[j] = -1
        
        Mi = np.zeros((nK + 3 * nS, 3 * nK + 1), dtype=DTYPE_F)
        Ni = np.zeros((nK + 3 * nS, nK), dtype=DTYPE_F)

        set_ls_matrices(grid, Mi, Ni, i, nK, nS, KSetv, Sv)

        if is_neumann:
            pass

        M = np.linalg.inv(Mi.T @ Mi) @ (Mi.T @ Ni)
        
        w_total = len(M[-1, :])

        if is_neumann:
            w_total = w_total - 1

        for j in range(w_total):
            weights[i, j] = M[-1, j]
            
        if is_neumann:
            neumann_ws[i] = M[-1, -1]

cdef void set_ls_matrices(Grid grid, 
                          cnp.ndarray Mi, cnp.ndarray Ni, 
                          const int v, const int nK, const int nS,
                          const DTYPE_F_t[::1] KSetv, const DTYPE_F_t[::1] Sv):
    
    cdef:
        int i, j, k
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] xv  = grid.point_coords[v]
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] xK  = np.zeros((nK, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] dKv = np.zeros((nK, 3), dtype=DTYPE_F)

        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] Ksetv_range = np.arange(nK)

    for i in range(nK):
        xK[i] = grid.centroid[KSetv[i]]
        dKv[i] = xK[i] - xv
    
    Mi[Ksetv_range, 3 * Ksetv_range] = dKv[:, 0]
    Mi[Ksetv_range, 3 * Ksetv_range + 1] = dKv[:, 1]
    Mi[Ksetv_range, 3 * Ksetv_range + 2] = dKv[:, 2]
    Mi[Ksetv_range, 3 * nK] = 1.0

    Ni[Ksetv_range, Ksetv_range] = 1.0

    
