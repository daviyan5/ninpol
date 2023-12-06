import pytest
import interpolator.grid
import numpy as np
import time

def create_esup(connectivity, n_elems, n_points_per_elem, n_points):
    
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
    
    return esup, esup_ptr, psup, psup_ptr

def test_grid():
   
    n_tests = 10
    n_repeats = 10
    avg_speedup = 0
    nelems_sp = np.linspace(100, 1e5, n_tests, dtype=np.int32)
    div_sp = np.linspace(9, 3, n_tests, dtype=np.int32)
    for case in range(1, n_tests):
        
        n_elems = nelems_sp[case]
        n_points_per_elem = np.ceil(np.sqrt(case) + 2).astype(int)
        div = div_sp[case]
        n_points = (n_elems * n_points_per_elem) // div
        print(f"[{str(case).zfill(2)}] \t n_elems: {n_elems}, n_points_per_elem: {n_points_per_elem}, n_points: {n_points}")
        grid = interpolator.grid.Grid(2, n_elems, n_points, n_points_per_elem)
        rand_array = np.zeros((n_elems, n_points_per_elem), dtype=np.int32).flatten()
        # Repeat np.arange(n_points) div times
        rand_array[:n_points * div] = np.tile(np.arange(n_points), div)
        # Shuffle the array
        np.random.shuffle(rand_array)
        rand_array = rand_array.reshape((n_elems, n_points_per_elem)).astype(np.int64)
        local_avg = 0
        esup, esup_ptr, psup, psup_ptr = None, None, None, None
        for j in range(n_repeats):
            start = time.time()
            grid.build(rand_array)
            end = time.time()
            elapsed_cpnp = end - start
            
            start = time.time()
            esup, esup_ptr, psup, psup_ptr = create_esup(rand_array, n_elems, n_points_per_elem, n_points)
            end = time.time()
            elapsed_pynp = end - start

            speedup = round(elapsed_pynp/elapsed_cpnp, 3)
            local_avg += speedup
        
        print(f"[{str(case).zfill(2)}] \t Cython Speedup: {local_avg / n_repeats} times")
        avg_speedup += local_avg / n_repeats
        print(f"[{str(case).zfill(2)}] \t Average Speedup: {round(avg_speedup/case, 3)} times")
        print()

        assert grid.esup is not None,                       "The esup array cannot be None."
        assert grid.esup_ptr is not None,                   "The esup_ptr array cannot be None."
        assert grid.esup_ptr.shape == (n_points + 1, ),     "The esup_ptr array has the wrong shape."

        assert grid.psup is not None,                       "The psup array cannot be None."
        assert grid.psup_ptr is not None,                   "The psup_ptr array cannot be None."
        assert grid.psup_ptr.shape == (n_points + 1, ),     "The psup_ptr array has the wrong shape."
        
        assert np.all(grid.esup_ptr == esup_ptr),           "The esup_ptr array is incorrect."
        assert np.all(grid.esup == esup),                   "The esup array is incorrect."
        
        
        assert np.all(grid.psup_ptr == psup_ptr),           "The psup_ptr array is incorrect."
        assert np.all(np.sort(grid.psup) == np.sort(psup)), "The psup array is incorrect."
        



