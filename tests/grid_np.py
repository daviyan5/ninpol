"""
This is a python/numpy mirror of "grid.pyx" file, used for performance and correctness testing.
"""

import numpy as np
import numba as nb

@nb.njit()
def build(connectivity, n_elems, n_points_per_elem, n_points):

    # Check that the connectivity matrix is not None and has the correct shape
    if connectivity is None:
        raise ValueError("The connectivity matrix cannot be None.")
    if connectivity.shape[0] != n_elems:
        raise ValueError("The number of rows in the connectivity matrix must be equal to the number of elements.")
    if connectivity.shape[1] != n_points_per_elem:
        raise ValueError("The number of columns in the connectivity matrix must be equal to the number of points per element.")
    
    
    esup_ptr = np.bincount(connectivity.ravel() + 1, minlength=n_points + 1)
    esup_ptr = np.cumsum(esup_ptr)
    esup = np.zeros(esup_ptr[n_points], dtype=np.int32)

    for i in range(n_elems):
        for j in range(n_points_per_elem):
            esup[esup_ptr[connectivity[i, j]]] = i
            esup_ptr[connectivity[i, j]] += 1
        
    for i in range(n_points, 0, -1):
        esup_ptr[i] = esup_ptr[i-1]

    esup_ptr[0] = 0

    psup_ptr = np.zeros(n_points + 1, dtype=np.int32)
    psup = np.zeros(esup_ptr[n_points] * (n_points_per_elem - 1), dtype=np.int32)
    stor_ptr = 0
    for i in range(n_points):
        elems = esup[esup_ptr[i]:esup_ptr[i+1]]
        if elems.shape[0] == 0:
            continue
        x = connectivity[elems, :].flatten()
        mx    = np.max(x) + 1
        used  = np.zeros(mx,dtype=np.uint8)
        used[x] = 1
        points = np.argwhere(used == 1)[:,0]
        points = points[points != i]
        psup[stor_ptr:stor_ptr + points.shape[0]] = points
        psup_ptr[i+1] = psup_ptr[i] + points.shape[0]
        stor_ptr += points.shape[0]
    psup = psup[:psup_ptr[-1]]
    return esup, esup_ptr, psup, psup_ptr
    