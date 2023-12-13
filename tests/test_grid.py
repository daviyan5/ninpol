import pytest
import numpy as np
import numba as nb
import warnings
import time
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

import ninpol.grid
import grid_np

# Test parameters

MIN_ELEM    = 5
MAX_ELEM    = 5e4
n_tests     = 15
n_repeats   = 3
linear      = False

# Test arrays
n_elems_sp  = np.linspace(MIN_ELEM, MAX_ELEM, n_tests, dtype=np.int32) if linear else np.logspace(np.log10(MIN_ELEM), np.log10(MAX_ELEM), n_tests, dtype=np.int32)
div_sp      = np.linspace(3, 10, n_tests, dtype=np.int32)



class TestGrid:
    @pytest.fixture(autouse=True)
    def set_parameters(self, request):
        global MIN_ELEM, MAX_ELEM, n_tests, n_repeats, linear, n_elems_sp, div_sp
        MIN_ELEM    = int(request.config.getoption("--min-elem", default=MIN_ELEM))
        MAX_ELEM    = int(request.config.getoption("--max-elem", default=MAX_ELEM))
        n_tests     = int(request.config.getoption("--n-test", default=n_tests))
        n_repeats   = int(request.config.getoption("--n-repeats", default=n_repeats))
        linear      = bool(request.config.getoption("--linear", default=linear))
        n_elems_sp  = np.linspace(MIN_ELEM, MAX_ELEM, n_tests, dtype=np.int32) if linear else np.logspace(np.log10(MIN_ELEM), np.log10(MAX_ELEM), n_tests, dtype=np.int32)
        div_sp      = np.linspace(3, 10, n_tests, dtype=np.int32)

    def test_imports(self):
        from ninpol import Grid
        from ninpol import Interpolator
    def test_grid_build(self):
        
        """
        Tests wether the grid is built correctly.
        """
        print(f"\n{Fore.CYAN}Testing correctness ========================================{Style.RESET_ALL}")
        print("{}{:<5} {:<15} {:<15} {:<15} {:<5}{}".format(Fore.GREEN, 
                                                            "Idx", "Nº of Elements", "Points/Element", "Nº of Points", "Status",
                                                            Style.RESET_ALL))
        for case in range(n_tests):
            n_elems           = n_elems_sp[case]
            n_points_per_elem = np.ceil(np.sqrt(case) + 2).astype(int)
            div               = div_sp[case]
            n_points          = n_elems * n_points_per_elem // div

            zfill_len = int(np.ceil(np.log10(n_tests)))
            index_str = "["+str(case).zfill(zfill_len)+"]"
            temp_str = "Building matrix..."
            print(f"{Fore.YELLOW}{index_str:<5} {temp_str:<15} {Style.RESET_ALL}", end="\r")
            rand_array = np.zeros((n_elems, n_points_per_elem), dtype=np.int32).flatten()
            rand_array[:n_points * div] = np.tile(np.arange(n_points), div)
            rand_array = rand_array.reshape((n_elems, n_points_per_elem)).astype(np.int64)
            rand_array = np.ascontiguousarray(rand_array)
            temp_str = "Building grid..."
            print(f"{Fore.YELLOW}{index_str:<5} {temp_str:<15} {Style.RESET_ALL}", end="\r")
            esup, esup_ptr, psup, psup_ptr = grid_np.build(rand_array, n_elems, n_points_per_elem, n_points)

            grid = ninpol.grid.Grid(2, n_elems, n_points, n_points_per_elem)
            grid.build(rand_array)
            
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
            print(f"{index_str:<5} {n_elems:<15} {n_points_per_elem:<15} {n_points:<15}", end="")
            status = "OK"
            print(f" {Fore.GREEN}{status:<5}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}============================================================={Style.RESET_ALL}")
    
    def test_grid_speed(self):
        """
        Tests wether the grid is built efficiently.
        """
        print(f"\n{Fore.CYAN}Testing speed ------------------------------------------------------------------------------{Style.RESET_ALL}")
        print(f"{Fore.LIGHTWHITE_EX}", end="")
        print("{}{:<5} {:<15} {:<15} {:<15} {:<15} {:<15} {:<5}{}".format(Fore.GREEN,
                                                                          "Idx", "Nº of Elements", "Points/Element", "Nº of Points", "Speedup", "Global Speedup", "Status",
                                                                          Style.RESET_ALL))
        print(f"{Style.RESET_ALL}", end="")
        global_avg = 0.
        first_call = True
        suboptimal = []
        for case in range(n_tests):
            n_elems           = n_elems_sp[case]
            n_points_per_elem = np.ceil(np.sqrt(case) + 2).astype(int)
            div               = div_sp[case]
            n_points          = n_elems * n_points_per_elem // div

            
            rand_array = np.zeros((n_elems, n_points_per_elem), dtype=np.int32).flatten()
            rand_array[:n_points * div] = np.tile(np.arange(n_points), div)
            np.random.shuffle(rand_array)
            rand_array = rand_array.reshape((n_elems, n_points_per_elem)).astype(np.int64)
            rand_array = np.ascontiguousarray(rand_array)
            if first_call:
                esup, esup_ptr, psup, psup_ptr = grid_np.build(rand_array, n_elems, n_points_per_elem, n_points)
                first_call = False
            grid = ninpol.grid.Grid(2, n_elems, n_points, n_points_per_elem)

            local_avg = 0.
            for rep in range(n_repeats):
                zfill_len = int(np.ceil(np.log10(n_tests)))
                rep_str = "["+str(rep).zfill(zfill_len)+"]"
                temp_str = "Cur Speedup:"
                print(f"{Fore.YELLOW}{rep_str:<5} {temp_str:<15} {round(local_avg/(rep + 1), 2)} {Style.RESET_ALL}", end="\r")
                start = time.time()
                grid.build(rand_array)
                end = time.time()
                elapsed_cpnp = end - start
                
                start = time.time()
                esup, esup_ptr, psup, psup_ptr = grid_np.build(rand_array, n_elems, n_points_per_elem, n_points)
                end = time.time()
                elapsed_pynp = end - start

                local_avg +=  elapsed_pynp / elapsed_cpnp

            local_speedup = local_avg / n_repeats
            global_avg += local_speedup
            global_speedup = global_avg / (case + 1)

            local_speedup = np.round(local_speedup, 2)
            global_speedup = np.round(global_speedup, 2)

            zfill_len = int(np.ceil(np.log10(n_tests)))
            index_str = "["+str(case).zfill(zfill_len)+"]"
            print(f"{index_str:<5} {n_elems:<15} {n_points_per_elem:<15} {n_points:<15} {local_speedup:<15} {global_speedup:<15}", end="")

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

            status = "FAST!" if local_speedup > 10. else "OK" if local_speedup > 5. else "SLOW" if local_speedup > 1. else "SLOW!!"
            print(f" {Fore.GREEN if local_speedup > 10. else Fore.LIGHTGREEN_EX if local_speedup > 5. else Fore.YELLOW if local_speedup > 1. else Fore.RED}{status:<5}{Style.RESET_ALL}")

            if local_speedup <= 2.:
                suboptimal.append(case)


        print(f"{Fore.CYAN}-------------------------------------------------------------------------------------------{Style.RESET_ALL}")

            
            



