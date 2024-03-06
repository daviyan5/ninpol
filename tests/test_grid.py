import pytest
import numpy as np

import time
import os
import sys
from colorama import Fore, Style

# Test parameters
n_tests     = 3
linear      = False
mesh_dir    = "tests/test-mesh/"


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class TestGrid:
    @pytest.fixture(autouse=True)
    def set_parameters(self, request):
        pass        

    def test_imports(self):
        from ninpol.mesh import Grid
        from ninpol import Interpolator
                    
    def test_grid_build(self):
        
        """
        Tests whether the grid is built correctly.
        """
        import ninpol
        import meshio
        import os

        # Iterate over n_tests files of mesh_dir
        files = sorted(os.listdir(mesh_dir))[1:]
        interpolador = ninpol.Interpolator()

        for case in range(n_tests):
            blockPrint()
            interpolador.load_mesh(mesh_dir + files[case])
            enablePrint()

            
    
    def test_grid_speed(self):
        """
        Tests wether the grid is built efficiently.
        """
        import ninpol
        import meshio
        import os
        # Iterate over n_tests files of mesh_dir
        files = sorted(os.listdir(mesh_dir))[1:]
        interpolador = ninpol.Interpolator()
        print("\n==============================================\n", end="")
        for case in range(n_tests):
            
            start_time = time.time()
            blockPrint()
            msh = meshio.read(mesh_dir + files[case])
            enablePrint()
            end_time = time.time()
            time_to_load = end_time - start_time
            
            start_time = time.time()
            args = interpolador.process_mesh(msh)
            end_time = time.time()

            time_to_process = end_time - start_time
            
            start_time = time.time()
            grid_obj = ninpol.Grid(args[0], 
                                   args[1], args[2], args[3], 
                                   args[4], args[5], args[6],
                                   args[7], args[8])
            grid_obj.build(args[9], args[10], args[11])
            end_time = time.time()

            time_to_build = end_time - start_time

            start_time = time.time()
            blockPrint()
            interpolador.load_mesh(mesh_dir + files[case])
            enablePrint()
            end_time = time.time()

            time_to_load_process_build = end_time - start_time
            color = [
                Fore.GREEN, 
                Fore.GREEN,
                Fore.GREEN
            ]
            mn = min(time_to_load, time_to_process, time_to_build)
            mx = max(time_to_load, time_to_process, time_to_build)
            
            color[0] = Fore.GREEN if time_to_load == mn else Fore.RED if time_to_load == mx else Fore.YELLOW
            color[1] = Fore.GREEN if time_to_process == mn else Fore.RED if time_to_process == mx else Fore.YELLOW
            color[2] = Fore.GREEN if time_to_build == mn else Fore.RED if time_to_build == mx else Fore.YELLOW

            print(f"{'Mesh:':<35} {Fore.LIGHTGREEN_EX}{files[case]}{Style.RESET_ALL}")
            print(f"{'Time to load mesh:':<35} {color[0]}{time_to_load:.2e} s{Style.RESET_ALL}")
            print(f"{'Time to process mesh:':<35} {color[1]}{time_to_process:.2e} s{Style.RESET_ALL}")
            print(f"{'Time to build grid:':<35} {color[2]}{time_to_build:.2e} s{Style.RESET_ALL}")
            print(f"{'Time to load, process and build:':<35} {time_to_load_process_build:.2e} s")
            print("==============================================")

                    
            



