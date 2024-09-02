import numpy as np
import scipy as sp
import time
import re

from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from libc.stdio cimport printf
from libc.math cimport sqrt

DTYPE_I = np.int64
DTYPE_F = np.float64

cdef:
    Logger logger = Logger("GLS")
    int logging = False

cdef void GLS(Grid grid, 
              const DTYPE_I_t[::1] in_points, 
              const DTYPE_I_t[::1] nm_points, 
              const DTYPE_F_t[:, :, ::1] permeability, 
              const DTYPE_F_t[::1] diff_mag,
              const DTYPE_F_t[::1] gN,
              DTYPE_F_t[:, ::1] weights, 
              DTYPE_F_t[::1] neumann_ws):
    """
    Main GLS function that processes the grid structure into substructures.
    """
    
    if (len(in_points) > 0):
        interpolate_nodes(grid, in_points, permeability, 0, gN, diff_mag, weights, neumann_ws)
    if (len(nm_points) > 0):
        interpolate_nodes(grid, nm_points, permeability, 1, gN, diff_mag, weights, neumann_ws)
    
    

cdef void interpolate_nodes(Grid grid, 
                            const DTYPE_I_t[::1] target,                # Input
                            const DTYPE_F_t[:, :, ::1] permeability,    # Input
                            const int is_neumann,                       # Input
                            const DTYPE_F_t[::1] gN,                    # Input
                            const DTYPE_F_t[::1] diff_mag,              # Input
                            DTYPE_F_t[:, ::1] weights,                  # Output
                            DTYPE_F_t[::1] neumann_ws,                  # Output
                            ):
    
    cdef: 
        dict log_dict = {}

    cdef:
        int i, j, k
        int point, bpoint
        int n_target = target.shape[0]
        int el_idx, fc_idx, fcb_idx
        int nK, nS, nb
        int sum_nL

        float neumann_val
        int n_pofa

        int M_size, w_total

        int MX_INDEX = max(grid.n_elems, grid.n_faces)

        DTYPE_F_t[:, ::1] normal_faces = grid.normal_faces
    
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

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] nL = np.zeros((grid.MX_FACES_PER_POINT, 3), dtype=DTYPE_F)
        
        
        cnp.ndarray[DTYPE_F_t, ndim=2] M = np.zeros((1, 1), dtype=DTYPE_F)

    setting_times  = 0.
    solvings_times = 0.

    start_time = time.time()
    for i, point in enumerate(target):
        el_idx = 0
        for j in range(grid.esup_ptr[point], grid.esup_ptr[point + 1]):
            KSetv[el_idx] = grid.esup[j]
            el_idx = el_idx + 1

        nK = el_idx
        for j in range(el_idx, grid.MX_ELEMENTS_PER_POINT):
            KSetv[j] = MX_INDEX

        fc_idx = 0
        for j in range(grid.fsup_ptr[point], grid.fsup_ptr[point + 1]):
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
        Ni = np.zeros((nK + 3 * nS + nb, nK + is_neumann), dtype=DTYPE_F)
        
        log_dict[point] = {}
        set_time = time.time()
        set_ls_matrices(grid, logger, point, nK, nS, KSetv, Sv, permeability, diff_mag, Mi, Ni, log_dict)
        setting_times += time.time() - set_time

        if is_neumann:
            neu_rows = np.arange(start=nK + 3 * nS, stop=nK + 3 * nS + nb)  
            
                        
            for j in range(nb):
                Ks_Svb[j]           = grid.esuf[grid.esuf_ptr[Svb[j]]]
                nL[j]               = np.dot(normal_faces[Svb[j]], permeability[Ks_Svb[j]])
                # Calculate neumann of the face
                n_pofa = 0
                neumann_val = 0
                for k, bpoint in enumerate(grid.inpofa[Svb[j]]):
                    if bpoint == -1:
                        break
                    neumann_val += gN[bpoint]
                    n_pofa += 1
                neumann_val = neumann_val / n_pofa
                Ni[neu_rows[j], nK] = neumann_val
                
            sorter = np.argsort(KSetv)                                                           
            Ik = sorter[np.searchsorted(KSetv, Ks_Svb, sorter=sorter)]

            Mi[neu_rows, 3 * Ik]     = -nL[:len(neu_rows), 0]                                                     
            Mi[neu_rows, 3 * Ik + 1] = -nL[:len(neu_rows), 1]                                                 
            Mi[neu_rows, 3 * Ik + 2] = -nL[:len(neu_rows), 2]                                        

        solving_time = time.time()
        M = sp.linalg.lstsq(Mi, Ni, cond=None, lapack_driver="gelsy")[0] 
        solvings_times += time.time() - solving_time

        M_size  = len(M)

        w_total = len(M[M_size - 1, :])
        
        if is_neumann:
            w_total = w_total - 1

        for j in range(w_total):
            weights[point, j] = M[M_size - 1, j]

        if is_neumann:
            neumann_ws[point] = M[M_size - 1, w_total - 1]

        if logging:
            logger.log("Interpolating point: %d" % point, "INFO")
            if nK + nS < 1000:
                temp_dict = {
                    "nK" : nK,
                    "nS" : nS,
                    "nb" : nb,
                    "neumann" : is_neumann,
                    "Mi" : np.asarray(Mi),
                    "Ni" : np.asarray(Ni),
                    "neu_rows" : np.asarray(neu_rows),
                    "Ik" : np.asarray(Ik),
                    "sorter" : np.asarray(sorter),
                    "Ks_Svb" : np.asarray(Ks_Svb),
                    "nL" : np.asarray(nL),
                }
            else:
                temp_dict = {
                    "nK" : nK,
                    "nS" : nS,
                    "nb" : nb,
                    "neumann" : is_neumann,
                    "Mi" : {"shape": np.shape(Mi)},
                    "Ni" : {"shape": np.shape(Ni)},
                    "neu_rows" : {"shape": np.shape(neu_rows)},
                    "Ik" : {"shape": np.shape(Ik)},
                    "sorter" : {"shape": np.shape(sorter)},
                    "Ks_Svb" : {"shape": np.shape(Ks_Svb)},
                    "nL" : {"shape": np.shape(nL)}
                }
            for key in temp_dict:
                log_dict[point][key] = temp_dict[key]
            logger.json(str(point), log_dict[point])

    
    print(f"Total time: {time.time() - start_time}")
    print(f"Setting time: {setting_times}")
    print(f"Solving time: {solvings_times}")

cdef void set_ls_matrices(Grid grid, Logger logger,
                          const int v, const int nK, const int nS,
                          const DTYPE_I_t[::1] KSetv, const DTYPE_I_t[::1] Sv, 
                          const DTYPE_F_t[:, :, ::1] permeability, const DTYPE_F_t[::1] diff_mag,
                          cnp.ndarray Mi, cnp.ndarray Ni, dict log_dict = {}):
    
    cdef:
        int i, j, k
        int curKSv, nKsv = 0
        cnp.ndarray[DTYPE_F_t, ndim=1, mode='c'] xv  = np.asarray(grid.point_coords[v])
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] xK  = np.zeros((nK, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] dKv = np.zeros((nK, 3), dtype=DTYPE_F)

        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] Ksetv_range = np.arange(nK)

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] xS = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] N_sj = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=1, mode='c'] eta_j = np.zeros(nS, dtype=DTYPE_F)
        cnp.ndarray[DTYPE_I_t, ndim=2, mode='c'] Ks_Sv = np.zeros((nS, 2), dtype=DTYPE_I)

        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] Ij1 = np.zeros(nS, dtype=DTYPE_I)
        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] Ij2 = np.zeros(nS, dtype=DTYPE_I)

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] T_sj1 = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] T_sj2 = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=1, mode='c'] tau_j2 = np.zeros((nS), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] tau_tsj2 = np.zeros((nS, 3), dtype=DTYPE_F)

        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] nL1 = np.zeros((nS, 3), dtype=DTYPE_F)
        cnp.ndarray[DTYPE_F_t, ndim=2, mode='c'] nL2 = np.zeros((nS, 3), dtype=DTYPE_F)

        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] idx1 = np.zeros(3 * nS - 2, dtype=DTYPE_I)
        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] idx2 = np.zeros(3 * nS - 2, dtype=DTYPE_I)
        cnp.ndarray[DTYPE_I_t, ndim=1, mode='c'] idx3 = np.zeros(3 * nS - 2, dtype=DTYPE_I)

        dict temp_dict = {}

    for i in range(nK):
        xK[i] = grid.centroids[KSetv[i]]
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
        
        eta_j[nKsv]    = max(diff_mag[Ks_Sv[nKsv, 0]], diff_mag[Ks_Sv[nKsv, 1]])
        T_sj1[nKsv]    = np.array([xv[0] - xS[i][0], 
                                   xv[1] - xS[i][1], 
                                   xv[2] - xS[i][2]])
        
        T_sj2[nKsv]    = np.cross(N_sj[i], T_sj1[nKsv])                             
        tau_j2[nKsv]   = np.linalg.norm(T_sj2[nKsv]) ** (-eta_j[nKsv])            

        
        
        tau_tsj2[nKsv] = np.array([tau_j2[nKsv] * T_sj2[nKsv, 0], 
                                   tau_j2[nKsv] * T_sj2[nKsv, 1], 
                                   tau_j2[nKsv] * T_sj2[nKsv, 2]])

        nL1[nKsv]      = np.dot(N_sj[i], permeability[Ks_Sv[nKsv, 0]])              
        nL2[nKsv]      = np.dot(N_sj[i], permeability[Ks_Sv[nKsv, 1]])              
        nKsv = nKsv + 1
    
    Ks_Sv = Ks_Sv[:nKsv]
    eta_j = eta_j[:nKsv]
    T_sj1 = T_sj1[:nKsv]
    T_sj2 = T_sj2[:nKsv]
    tau_j2 = tau_j2[:nKsv]
    tau_tsj2 = tau_tsj2[:nKsv]
    nL1 = nL1[:nKsv]
    nL2 = nL2[:nKsv]
    sorter = np.argsort(KSetv)              

    Ij1 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 0], sorter=sorter)]    # Index of the volumes around the point (KSetv) that have the face Sv. Basically, i such that KSetv[i] has face[j] in its faces
    Ij2 = sorter[np.searchsorted(KSetv, Ks_Sv[:, 1], sorter=sorter)]    # Same above
    


    idx1 = np.arange(start=nK,     stop=nK + 3 * nS - 2, step=3)
    idx2 = np.arange(start=nK + 1, stop=nK + 3 * nS - 1, step=3)
    idx3 = np.arange(start=nK + 2, stop=nK + 3 * nS,     step=3)

    idx1 = idx1[:nKsv]
    idx2 = idx2[:nKsv]
    idx3 = idx3[:nKsv]

    if logging:
        if nK + nS < 1000:
            temp_dict = {
                "v" : v,

                "nK" : nK,
                "nS" : nS,

                "KSetv" : np.asarray(KSetv),
                "Sv" : np.asarray(Sv),

                "xv" : np.asarray(xv),
                "xK" : np.asarray(xK),
                "dKv": np.asarray(dKv),

                "KSetv_range": np.asarray(KSetv),
                "xS" : np.asarray(xS),
                "N_sj": np.asarray(N_sj),

                "eta_j": np.asarray(eta_j),
                "Ks_Sv": np.asarray(Ks_Sv),

                "Ij1": np.asarray(Ij1),
                "Ij2": np.asarray(Ij2),

                "T_sj1": np.asarray(T_sj1),
                "T_sj2": np.asarray(T_sj2),

                "tau_j2": np.asarray(tau_j2),
                "tau_tsj2": np.asarray(tau_tsj2),

                "nL1": np.asarray(nL1),
                "nL2": np.asarray(nL2),

                "idx1": np.asarray(idx1),
                "idx2": np.asarray(idx2),
                "idx3": np.asarray(idx3)
            }
        else:
            temp_dict = {
                "v" : v,
                "nK" : nK,
                "nS" : nS,

                "KSetv" : {"shape": np.shape(KSetv)},
                "Sv" : {"shape": np.shape(Sv)},
                "xv" : {"shape": np.shape(xv)},
                "xK" : {"shape": np.shape(xK)},
                "dKv": {"shape": np.shape(dKv)},
                "KSetv_range": {"shape": np.shape(KSetv)},
                "xS" : {"shape": np.shape(xS)},
                "N_sj": {"shape": np.shape(N_sj)},
                "eta_j": {"shape": np.shape(eta_j)},
                "Ks_Sv": {"shape": np.shape(Ks_Sv)},

                "Ij1": {"shape": np.shape(Ij1)},
                "Ij2": {"shape": np.shape(Ij2)},

                "T_sj1": {"shape": np.shape(T_sj1)},
                "T_sj2": {"shape": np.shape(T_sj2)},

                "tau_j2": {"shape": np.shape(tau_j2)},
                "tau_tsj2": {"shape": np.shape(tau_tsj2)},

                "nL1": {"shape": np.shape(nL1)},
                "nL2": {"shape": np.shape(nL2)},

                "idx1": {"shape": np.shape(idx1)},
                "idx2": {"shape": np.shape(idx2)},
                "idx3": {"shape": np.shape(idx3)}

            }

        # append the keys from temp_dict to log_dict[v]
        for key in temp_dict:
            log_dict[v][key] = temp_dict[key]
    try:
    
        Mi[idx1, 3 * Ij1] = -nL1[:, 0]                                                           # Setting x-component differences for first volumes in Mi.
        Mi[idx1, 3 * Ij1 + 1] = -nL1[:, 1]                                                       # Setting y-component differences for first volumes in Mi.
        Mi[idx1, 3 * Ij1 + 2] = -nL1[:, 2]                                                       # Setting z-component differences for first volumes in Mi.

        Mi[idx1, 3 * Ij2] = nL2[:, 0]                                                            # Setting x-component differences for second volumes in Mi.
        Mi[idx1, 3 * Ij2 + 1] = nL2[:, 1]                                                        # Setting y-component differences for second volumes in Mi.
        Mi[idx1, 3 * Ij2 + 2] = nL2[:, 2]                                                        # Setting z-component differences for second volumes in Mi.

        Mi[idx2, 3 * Ij1] = -T_sj1[:, 0]                                                         # Setting x-component differences in Mi.
        Mi[idx2, 3 * Ij1 + 1] = -T_sj1[:, 1]                                                     # Setting y-component differences in Mi.
        Mi[idx2, 3 * Ij1 + 2] = -T_sj1[:, 2]                                                     # Setting z-component differences in Mi.

        Mi[idx2, 3 * Ij2] = T_sj1[:, 0]                                                          # Setting x-component differences in Mi.
        Mi[idx2, 3 * Ij2 + 1] = T_sj1[:, 1]                                                      # Setting y-component differences in Mi.
        Mi[idx2, 3 * Ij2 + 2] = T_sj1[:, 2]                                                      # Setting z-component differences in Mi.

        Mi[idx3, 3 * Ij1] = -tau_tsj2[:, 0]                                                      # Setting x-component differences in Mi.
        Mi[idx3, 3 * Ij1 + 1] = -tau_tsj2[:, 1]                                                  # Setting y-component differences in Mi.
        Mi[idx3, 3 * Ij1 + 2] = -tau_tsj2[:, 2]                                                  # Setting z-component differences in Mi.

        Mi[idx3, 3 * Ij2] = tau_tsj2[:, 0]                                                       # Setting x-component differences in Mi.
        Mi[idx3, 3 * Ij2 + 1] = tau_tsj2[:, 1]                                                   # Setting y-component differences in Mi.
        Mi[idx3, 3 * Ij2 + 2] = tau_tsj2[:, 2] 
        
    except Exception as e:
        logger.log(f"Error setting LS matrices for point {v}: {e}", "ERROR")
        logger.json(str(v), log_dict[v])
        raise e
    