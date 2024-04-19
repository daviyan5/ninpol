import pytest
import numpy as np
import ninpol
import meshio
import timeit
import os
import sys
import yaml
from colorama import Fore, Style
import memory_profiler
from results import test_graphs

# Test parameters
mesh_dir    = "tests/utils/altered_mesh/"
n_number    = 1
n_repeats   = 1

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
        from ninpol import Grid
        from ninpol import Interpolator
                    
    def test_grid_build(self):
        """
        Tests whether the grid is built correctly.
        """

        # Iterate over n_files files of mesh_dir
        files   = sorted(os.listdir(mesh_dir))
        # Remove .gitkeep
        if ".gitkeep" in files:
            files.remove(".gitkeep")
        n_files = len(files)
        
        interpolador = ninpol.Interpolator()

        for case in range(n_files):
            blockPrint()
            interpolador.load_mesh(mesh_dir + files[case])
            enablePrint()

    def test_grid_speed(self):
        """
        Tests whether the grid is built efficiently.
        """
    
        def load_mesh(mesh_dir, filename):
            msh = meshio.read(os.path.join(mesh_dir, filename))

        def process_mesh(interpolador, msh):
            args = interpolador.process_mesh(msh)

        def build_grid(interpolador, args):
            grid_obj = ninpol.Grid(*args[:9])
            grid_obj.build(*args[9:])

        def load_process_build(interpolador, mesh_dir, filename):
            interpolador.load_mesh(os.path.join(mesh_dir, filename))

        def interpolate(interpolador, points):
            interpolador.interpolate(points, "linear", "inv_dist", "ctp")
        
        def average_memory(func):
            average = 0
            return 0
            for i in range(n_tests):
                current_usage = memory_profiler.memory_usage(-1, interval=0.2, timeout=2, max_usage=True)
                memory_consumption = memory_profiler.memory_usage(func, interval=0.0001, max_usage=True)
                memory_consumption -= current_usage
                average += memory_consumption / n_tests

            return average
        
        files   = sorted(os.listdir(mesh_dir))
        # Remove .gitkeep
        if ".gitkeep" in files:
            files.remove(".gitkeep")
        n_files = len(files)
        
        print("\n==============================================\n")

        # Create dict for yaml with results
        result_dict = {}
        result_dict["n_files"] = n_files
        result_dict["files"] = {}
        
        for case in range(n_files):
            interpolador = ninpol.Interpolator()
            filename = files[case]

            # File size in MB
            filesize_b = os.path.getsize(os.path.join(mesh_dir, filename)) / (1024 * 1024)
            filesize = np.round(filesize_b, 3)

            time_to_load = min(timeit.repeat(lambda: load_mesh(mesh_dir, filename), number=n_number, repeat=n_repeats))
            memory_to_load = average_memory((load_mesh, (mesh_dir, filename)))

            msh = meshio.read(os.path.join(mesh_dir, filename))

            time_to_process = min(timeit.repeat(lambda: process_mesh(interpolador, msh), number=n_number, repeat=n_repeats))
            memory_to_process = average_memory((process_mesh, (interpolador, msh)))
            

            args = interpolador.process_mesh(msh)
            grid_obj = ninpol.Grid(*args[:9])
            grid_obj.build(*args[9:])

            time_to_build = min(timeit.repeat(lambda: build_grid(interpolador, args), number=n_number, repeat=n_repeats))
            memory_to_build = average_memory((build_grid, (interpolador, args)))

            time_to_load_process_build = min(timeit.repeat(lambda: load_process_build(interpolador, mesh_dir, filename), number=n_number, repeat=n_repeats))
            memory_to_load_process_build = average_memory((load_process_build, (interpolador, mesh_dir, filename)))

            points = np.arange(grid_obj.n_points)
            n_points = grid_obj.n_points

            time_to_interpolate = min(timeit.repeat(lambda: interpolate(interpolador, points), number=n_number, repeat=n_repeats))
            memory_to_interpolate = average_memory((interpolate, (interpolador, points)))

            
            time_color = [
                Fore.GREEN, 
                Fore.GREEN,
                Fore.GREEN,
                Fore.GREEN
            ]
            memory_color = [
                Fore.GREEN, 
                Fore.GREEN,
                Fore.GREEN,
                Fore.GREEN
            ]

            result_dict["files"][filename] = {
                "file_size": filesize_b,
                "n_points": n_points,
                "time_to_build": time_to_load_process_build,
                "time_to_interpolate": time_to_interpolate,
                "memory_to_build": memory_to_load_process_build,    
                "memory_to_interpolate": memory_to_interpolate
            }
            mn_time = min(time_to_load, time_to_process, time_to_build, time_to_interpolate)
            mx_time = max(time_to_load, time_to_process, time_to_build, time_to_interpolate)

            decide = lambda x: Fore.GREEN if x == mn_time else Fore.RED if x == mx_time else Fore.YELLOW
            time_color[0] = decide(time_to_load)
            time_color[1] = decide(time_to_process)
            time_color[2] = decide(time_to_build)
            time_color[3] = decide(time_to_interpolate)

            mn_memory = min(memory_to_load, memory_to_process, memory_to_build, memory_to_interpolate)
            mx_memory = max(memory_to_load, memory_to_process, memory_to_build, memory_to_interpolate)

            decide = lambda x: Fore.GREEN if x == mn_memory else Fore.RED if x == mx_memory else Fore.YELLOW
            memory_color[0] = decide(memory_to_load)
            memory_color[1] = decide(memory_to_process)
            memory_color[2] = decide(memory_to_build)
            memory_color[3] = decide(memory_to_interpolate)

            print(f"{'Case:':<35} {Fore.LIGHTBLUE_EX}{f'{case + 1}/{n_files}':<10}{Style.RESET_ALL}")
            print(f"{'Mesh:':<35} {Fore.LIGHTBLUE_EX}{files[case]:<10}{Style.RESET_ALL}")
            print(f"{'File Size':<35} {Fore.LIGHTBLUE_EX}{filesize} MB{Style.RESET_ALL}")
            print(f"{'Nº of points:':<35} {Fore.LIGHTBLUE_EX}{n_points:<10}{Style.RESET_ALL}")

            print()

            print(f"{'Time to load, process and build:':<35} {time_to_load_process_build:.2e} s{Style.RESET_ALL}")
            print(f"{'Time to load mesh:':<35} {time_color[0]}{time_to_load:.2e} s{Style.RESET_ALL}")
            print(f"{'Time to process mesh:':<35} {time_color[1]}{time_to_process:.2e} s{Style.RESET_ALL}")
            print(f"{'Time to build grid:':<35} {time_color[2]}{time_to_build:.2e} s{Style.RESET_ALL}")
            print(f"{'Time to interpolate:':<35} {time_color[3]}{time_to_interpolate:.2e} s{Style.RESET_ALL}")

            # print()
            # print(f"{'Memory to load, process and build:':<35} {memory_to_load_process_build:.2e} MB{Style.RESET_ALL}")
            # print(f"{'Memory to load mesh:':<35} {memory_color[0]}{memory_to_load:.2e} MB{Style.RESET_ALL}")
            # print(f"{'Memory to process mesh:':<35} {memory_color[1]}{memory_to_process:.2e} MB{Style.RESET_ALL}")
            # print(f"{'Memory to build grid:':<35} {memory_color[2]}{memory_to_build:.2e} MB{Style.RESET_ALL}")
            # print(f"{'Memory to interpolate:':<35} {memory_color[3]}{memory_to_interpolate:.2e} MB{Style.RESET_ALL}")


            print("==============================================")

        with open("tests/results/performance_test.yaml", "w") as f:
            yaml.dump(result_dict, f)
        
        test_graphs.graph_performance()
                



