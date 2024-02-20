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

import read_msh

# Test parameters
n_repeats   = 12
linear      = False
test_dir    = "tests/test-mesh/"


class TestGrid:
    @pytest.fixture(autouse=True)
    def set_parameters(self, request):
        global n_repeats, linear
        n_repeats   = int(request.config.getoption("--n-repeats", default=n_repeats))
        linear      = bool(request.config.getoption("--linear", default=linear))
        

    def test_imports(self):
        from ninpol import Grid
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
        Tests wether the grid is built correctly.
        """
        print(f"\n{Fore.CYAN}Testing correctness ========================================{Style.RESET_ALL}")
        print("{}{:<5} {:<15} {:<15} {:<15} {:<5}{}".format(Fore.GREEN, 
                                                            "Idx", "Nº of Elements", "Points/Element", "Nº of Points", "Status",
                                                            Style.RESET_ALL))
        n_tests = 23
        for case in range(n_tests):
            # Load mesh and prepare data
            nodes_coords, matrix, elem_types = read_msh.read_msh_file(test_dir + f"test{case + 1}.msh")
            n_elems           = len(matrix)
            n_points_per_elem = len(matrix[0])
            n_points          = len(nodes_coords)
            zfill_len = int(np.ceil(np.log10(n_tests)))
            index_str = "[" + str(case).zfill(zfill_len) + "]"

            temp_str = "Building grid..."
            print(f"{Fore.YELLOW}{index_str:<5} {temp_str:<15} {Style.RESET_ALL}", end="\r")
            np_grid = grid_np.Grid(3, n_elems, n_points)
            np_grid.build(matrix, elem_types)

            grid = ninpol.grid.Grid(3, n_elems, n_points)
            grid.build(matrix, elem_types)

            self.compare_and_assert(np_grid, grid)

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
        n_tests = 23
        for case in range(n_tests):
            nodes_coords, matrix, elem_types = read_msh.read_msh_file(test_dir + f"test{case + 1}.msh")
            n_elems           = len(matrix)
            n_points_per_elem = len(matrix[0])
            n_points          = len(nodes_coords)

            
            if first_call:
                np_grid = grid_np.Grid(3, n_elems, n_points)
                np_grid.build(matrix, elem_types)
                first_call = False
            grid = ninpol.grid.Grid(3, n_elems, n_points)
            local_avg = 0.
            for rep in range(n_repeats):
                zfill_len = int(np.ceil(np.log10(n_tests)))
                rep_str = "["+str(rep).zfill(zfill_len)+"]"
                temp_str = "Cur Speedup:"
                print(f"{Fore.YELLOW}{rep_str:<5} {temp_str:<15} {round(local_avg/(rep + 1), 2)} {Style.RESET_ALL}", end="\r")
                
                start = time.time()
                
                grid.build(matrix, elem_types)

                end = time.time()

                elapsed_cpnp = end - start
                
                start = time.time()

                np_grid = grid_np.Grid(3, n_elems, n_points)
                np_grid.build(matrix, elem_types)

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

            self.compare_and_assert(np_grid, grid)

            status = "FAST!" if local_speedup > 10. else "OK" if local_speedup > 5. else "SLOW" if local_speedup > 1. else "SLOW!!"
            print(f" {Fore.GREEN if local_speedup > 10. else Fore.LIGHTGREEN_EX if local_speedup > 5. else Fore.YELLOW if local_speedup > 1. else Fore.RED}{status:<5}{Style.RESET_ALL}")

            if local_speedup <= 2.:
                suboptimal.append(case)


        print(f"{Fore.CYAN}-------------------------------------------------------------------------------------------{Style.RESET_ALL}")

# NOT USABLE
def generate_random_connectivity(self, n_elems, dim='3d'):
    import yaml
    with open("ninpol/utils/point_ordering.yaml", "r") as file:
        point_ordering = yaml.safe_load(file)
    max_points_per_elem = 0
    max_num_faces = 0
    for elem in point_ordering[dim]:
        max_points_per_elem = max(max_points_per_elem, point_ordering[dim][elem]['number_of_points'])
        
        max_num_faces = max(max_num_faces, len(point_ordering[dim][elem]['faces']))
    matrix = np.ones((n_elems, max_points_per_elem), dtype=np.int32) * -1
    visited_faces = np.zeros((n_elems, max_num_faces), dtype=bool)

    cur_elem_type = np.random.choice(list(point_ordering[dim].keys()))
    n_points = point_ordering[dim][cur_elem_type]['number_of_points']
    matrix[0, :n_points] = np.arange(n_points)
    cur_elem_count = 1

    queue = [(0, cur_elem_type)]
    print("Initial element: ", cur_elem_type)
    print("Initial queue: ", queue)
    print("Max points per elem: ", max_points_per_elem)
    print("Max num faces: ", max_num_faces)
    print()

    def get_face_index(face_list, face_type):
        for index, face in enumerate(face_list):
            if len(face) == face_type:
                return index
        return None
    aux_type = ["" for _ in range(n_elems)]
    while cur_elem_count < n_elems:
        cur_index, cur_elem_type = queue.pop(0)
        aux_type[cur_index] = cur_elem_type
        print("For cur index: ", cur_index, " and cur elem type: ", cur_elem_type)
        for face_index, face in enumerate(point_ordering[dim][cur_elem_type]['faces']):
            print("\t For face index: ", face_index, " and face: ", face, " of type: ", len(face), " and points: ", matrix[cur_index][face])
            if visited_faces[cur_index, face_index]:
                continue
            else:

                cur_elem_face_type = len(face)
                cur_elem_face_points = matrix[cur_index][face]
                
                next_elem_type = np.random.choice(list(point_ordering[dim].keys()))
                next_elem_index = cur_elem_count
                while True:
                    next_elem_face_index = get_face_index(point_ordering[dim][next_elem_type]['faces'], cur_elem_face_type)
                    if next_elem_face_index is not None and not visited_faces[next_elem_index, next_elem_face_index]:
                        break
                    else:
                        next_elem_type = np.random.choice(list(point_ordering[dim].keys()))
                print("\t\t Next elem type: ", next_elem_type, " and next elem face index: ", next_elem_face_index)
                next_elem_face = point_ordering[dim][next_elem_type]['faces'][next_elem_face_index]
                next_elem_face_points = cur_elem_face_points

                # If theres a -1 in next_elem_face_points, throw an error
                if -1 in next_elem_face_points:
                    raise ValueError("There's a -1 in the next_elem_face_points")
                
                matrix[next_elem_index][next_elem_face] = next_elem_face_points
                for j in range(point_ordering[dim][next_elem_type]['number_of_points']):
                    if matrix[next_elem_index][j] == -1:
                        matrix[next_elem_index][j] = np.max(matrix) + 1
                visited_faces[next_elem_index, next_elem_face_index] = True
                queue.append((next_elem_index, next_elem_type))
                cur_elem_count += 1
                aux_type[next_elem_index] = next_elem_type
                if cur_elem_count == n_elems:
                    break
    
    return matrix
                    
            



