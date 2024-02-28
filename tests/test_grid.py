import pytest
import numpy as np

import time
from colorama import Fore, Style

# Test parameters
n_repeats   = 12
linear      = False
mesh_dir    = "tests/test-mesh/"


class TestGrid:
    @pytest.fixture(autouse=True)
    def set_parameters(self, request):
        global n_repeats, linear
        n_repeats   = int(request.config.getoption("--n-repeats", default=n_repeats))
        linear      = bool(request.config.getoption("--linear", default=linear))
        

    def test_imports(self):
        from ninpol.mesh import Grid
        from ninpol import Interpolator

    
    def compare_and_assert(self, np_grid, grid):
        # Compare each non-function attribute between the two objects
        # Only the ones that are np.ndarray's
        wrong = []
        
        for attr in dir(np_grid):
            if not attr.startswith("__") and not callable(getattr(np_grid, attr)):
                np_attr = getattr(np_grid, attr)
                grid_attr = getattr(grid, attr)
                
                if isinstance(np_attr, np.ndarray):
                    grid_attr = np.sort(np.asarray(grid_attr).flatten())
                    np_attr = np.sort(np_attr.flatten())
                
                is_ok = np.allclose(np_attr, grid_attr)
                if not is_ok:
                    print(f"{Fore.RED} NOT OK for {attr} ! {Style.RESET_ALL}")
                    # Print the indexes of the items that are different
                    np_flat = np_attr.flatten()
                    grid_flat = grid_attr.flatten()
                    for i in range(len(np_flat)):
                        if np_flat[i] != grid_flat[i]:
                            print(f"{Fore.RED}np_flat[{i}] = {np_flat[i]} != grid_flat[{i}] = {grid_flat[i]}{Style.RESET_ALL}") 
                    wrong.append(attr)
        
        for attr in wrong:
            print(f"{Fore.RED}Attribute {attr} is wrong!{Style.RESET_ALL}")
        assert len(wrong) == 0
                    
    def test_grid_build(self):
        
        """
        Tests whether the grid is built correctly.
        """
        import ninpol
        import grid_np
        import meshio
        import os

        print(f"\n{Fore.CYAN}Testing correctness ========================================{Style.RESET_ALL}")
        print("{}{:<5} {:<15} {:<15} {:<15} {:<5}{}".format(Fore.GREEN, 
                                                            "Idx", "Nº of Elements", "Points/Element", "Nº of Points", "Status",
                                                            Style.RESET_ALL))
        n_tests = 3
        i = ninpol.Interpolator()

        # Iterate over n_tests files of mesh_dir
        files = sorted(os.listdir(mesh_dir))[1:]

        for case in range(n_tests):
            
            zfill_len = int(np.ceil(np.log10(n_tests)))
            index_str = "[" + str(case).zfill(zfill_len) + "]"

            temp_str = "Building grid..."
            print(f"{Fore.YELLOW}{index_str:<5} {temp_str:<15} {Style.RESET_ALL}", end="\r")
            msh = meshio.read(mesh_dir + files[case])
            args = i.process_mesh(msh)
            print(args)
            (dim, n_elems, n_points, n_faces, 
             nfael, lnofa, lpofa, 
             matrix, elem_types, n_points_per_elem) = args

            np_grid = grid_np.Grid(dim, n_elems, n_points, n_faces, nfael, lnofa, lpofa)
            np_grid.build(matrix, elem_types, n_points_per_elem)

            #grid = ninpol.grid.Grid(3, n_elems, n_points)
            #grid.build(matrix, elem_types)

            #self.compare_and_assert(np_grid, grid)

            #s\print(f"{index_str:<5} {n_elems:<15} {n_points_per_elem:<15} {n_points:<15}", end="")
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
        return 
        global_avg = 0.
        first_call = True
        suboptimal = []
        n_tests = 23
        for case in range(n_tests):
            

            
            if first_call:
                # np_grid = grid_np.Grid(3, n_elems, n_points)
                # np_grid.build(matrix, elem_types)
                first_call = False
            # grid = ninpol.grid.Grid(3, n_elems, n_points)
            local_avg = 0.
            for rep in range(n_repeats):
                zfill_len = int(np.ceil(np.log10(n_tests)))
                rep_str = "["+str(rep).zfill(zfill_len)+"]"
                temp_str = "Cur Speedup:"
                print(f"{Fore.YELLOW}{rep_str:<5} {temp_str:<15} {round(local_avg/(rep + 1), 2)} {Style.RESET_ALL}", end="\r")
                
                start = time.time()
                
                # grid.build(matrix, elem_types)

                end = time.time()

                elapsed_cpnp = end - start
                
                start = time.time()

                # np_grid = grid_np.Grid(3, n_elems, n_points)
                # np_grid.build(matrix, elem_types)

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
            # print(f"{index_str:<5} {n_elems:<15} {n_points_per_elem:<15} {n_points:<15} {local_speedup:<15} {global_speedup:<15}", end="")

            # self.compare_and_assert(np_grid, grid)

            status = "FAST!" if local_speedup > 10. else "OK" if local_speedup > 5. else "SLOW" if local_speedup > 1. else "SLOW!!"
            print(f" {Fore.GREEN if local_speedup > 10. else Fore.LIGHTGREEN_EX if local_speedup > 5. else Fore.YELLOW if local_speedup > 1. else Fore.RED}{status:<5}{Style.RESET_ALL}")

            if local_speedup <= 2.:
                suboptimal.append(case)


        print(f"{Fore.CYAN}-------------------------------------------------------------------------------------------{Style.RESET_ALL}")

                    
            



