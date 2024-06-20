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
                            const DTYPE_F_t[::1] target,                # Input
                            const DTYPE_F_t[:, :, ::1] permeability,    # Input
                            const int is_neumann,                       # Input
                            const DTYPE_F_t[::1] gN,                    # Input
                            const DTYPE_F_t[::1] diff_mag,              # Input
                            DTYPE_F_t[:, ::1] weights,                  # Output
                            DTYPE_F_t[::1] neumann_ws,                  # Output
                            ):
    
    cdef:
        int i, j, k
        int n_target = target.shape[0]
        int el_idx, fc_idx, fcb_idx
        int nK, nS, nb
        int sum_nL

        int MX_INDEX = max(grid.n_elems, grid.n_faces)

        DTYPE_F_t[:, ::1] normal_faces = grid.normal_faces
        
        DTYPE_F_t[:, ::1] nL = np.zeros((grid.MX_FACES_PER_POINT, 3), dtype=DTYPE_F)

        # Volumes surrounding the target point
        DTYPE_I_t[::1] KSetv  = np.zeros(grid.MX_ELEMENTS_PER_POINT, dtype=DTYPE_I)

        # Faces surrounding the target point
        DTYPE_I_t[::1] Sv     = np.zeros(grid.MX_FACES_PER_POINT, dtype=DTYPE_I)

        # Boundary faces surrounding the target point
        DTYPE_I_t[::1] Svb    = np.zeros(grid.MX_FACES_PER_POINT, dtype=DTYPE_I)
        
        # Pair of volumes surrounding the boundary face
        DTYPE_I_t[::1] Ks_Svb = np.zeros(grid.MX_FACES_PER_POINT, dtype=DTYPE_I)

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] Mi = np.zeros((1, 1), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] Ni = np.zeros((1, 1), dtype=DTYPE_F)
        
        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] neu_rows = np.zeros(1, dtype=DTYPE_I)
        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] Ik       = np.zeros(1, dtype=DTYPE_I)
        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] sorter   = np.zeros(1, dtype=DTYPE_I)
        
        
        DTYPE_F_t[:, ::1] M = np.zeros((1, 1), dtype=DTYPE_F)

    for i in range(n_target):
        el_idx = 0
        for j in range(grid.esup_ptr[i], grid.esup_ptr[i + 1]):
            KSetv[el_idx] = grid.esup[j]
            el_idx = el_idx + 1

        nK = el_idx
        for j in range(el_idx, grid.MX_ELEMENTS_PER_POINT):
            KSetv[j] = MX_INDEX

        fc_idx = 0
        for j in range(grid.fsup_ptr[i], grid.fsup_ptr[i + 1]):
            Sv[fc_idx] = grid.fsup[j]
            fc_idx = fc_idx + 1
        nS = fc_idx
        for j in range(fc_idx, grid.MX_FACES_PER_POINT):
            Sv[j] = MX_INDEX
        fcb_idx = 0
        for j in range(fc_idx):
            if grid.boundary_faces[Sv[j]] == 1:
                Svb[fcb_idx] = Sv[j]
                fcb_idx = fcb_idx + 1
        for j in range(fcb_idx, grid.MX_FACES_PER_POINT):
            Svb[j] = MX_INDEX
        nb = fcb_idx

        Mi = np.zeros((nK + 3 * nS + nb, 3 * nK + 1), dtype=DTYPE_F)
        Ni = np.zeros((nK + 3 * nS + nb, nK), dtype=DTYPE_F)

        set_ls_matrices(grid, i, nK, nS, KSetv, Sv, diff_mag, Mi, Ni)

        if is_neumann:
            neu_rows = np.arange(start=nK + 3 * nS, stop=nK + 3 * nS + nb)                       

            for i in range(nb):
                Ks_Svb[i]           = grid.esuf[grid.esuf_ptr[Svb[i]]]
                nL[i]               = np.dot(normal_faces[Svb[i]], permeability[Ks_Svb[i]])
                Ni[neu_rows[i], nK] = gN[Svb[i]]
                
            sorter = np.argsort(KSetv)                                                           
            Ik = sorter[np.searchsorted(KSetv, Ks_Svb, sorter=sorter)]

            for i in range(nb):
                Mi[neu_rows[i], 3 * Ik[i]]     = -nL[i, 0]                                               
                Mi[neu_rows[i], 3 * Ik[i] + 1] = -nL[i, 1]                                                 
                Mi[neu_rows[i], 3 * Ik[i] + 2] = -nL[i, 2]                                        

        M = np.linalg.inv(Mi.T @ Mi) @ (Mi.T @ Ni)
        
        w_total = len(M[-1, :])

        if is_neumann:
            w_total = w_total - 1

        for j in range(w_total):
            weights[i, j] = M[-1, j]
            
        if is_neumann:
            neumann_ws[i] = M[-1, -1]

cdef void set_ls_matrices(Grid grid, 
                          const int v, const int nK, const int nS,
                          const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, const DTYPE_F_t[::1] diff_mag,
                          cnp.ndarray Mi, cnp.ndarray Ni):
    
    cdef:
        int i, j, k
        int curKSv, nKsv
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] xv  = grid.point_coords[v]
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] xK  = np.zeros((nK, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] dKv = np.zeros((nK, 3), dtype=DTYPE_F)

        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] Ksetv_range = np.arange(nK)

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] xS = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] N_sj = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=1, mode='c'] eta_j = np.zeros(nS, dtype=DTYPE_F)
        cnp.ndarray[DTYPE_I_t, ndim=2, mode='c'] Ks_Sv = np.zeros((nS, 2), dtype=DTYPE_I)

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] Ij1 = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] Ij2 = np.zeros((nS, 3), dtype=DTYPE_F)


    for i in range(nK):
        xK[i] = grid.centroid[KSetv[i]]
        dKv[i] = xK[i] - xv
    
    Mi[Ksetv_range, 3 * Ksetv_range]     = dKv[:, 0]
    Mi[Ksetv_range, 3 * Ksetv_range + 1] = dKv[:, 1]
    Mi[Ksetv_range, 3 * Ksetv_range + 2] = dKv[:, 2]
    Mi[Ksetv_range, 3 * nK] = 1.0

    Ni[Ksetv_range, Ksetv_range] = 1.0

    for i in range(nS):
        xS[i] = grid.faces_centers[Sv[i]]
        N_sj[i] = grid.normal_faces[Sv[i]]
        curKSv = grid.esuf_ptr[Sv[i] + 1] - grid.esuf_ptr[Sv[i]]
        
        if curKSv < 2:
            continue
        Ks_Sv[nKsv, 0] = grid.esuf[grid.esuf_ptr[Sv[i]]]
        Ks_Sv[nKsv, 1] = grid.esuf[grid.esuf_ptr[Sv[i]] + 1]

        eta_j[nKsv] = np.max(diff_mag[Ks_Sv[nKsv, 0]], diff_mag[Ks_Sv[nKsv, 1]])
        nKsv = nKsv + 1

    Ks_Sv = Ks_Sv[:nKsv]
    sorter = np.argsort(KSetv)
    Ij1 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 0], sorter=sorter)]    # Index of the volumes around the point (KSetv) that have the face Sv. Basically, i such that KSetv[i] has face[j] in its faces
    Ij2 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 1], sorter=sorter)]    # Same above
    

    
