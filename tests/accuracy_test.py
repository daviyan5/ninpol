import pytest
import numpy as np
import ninpol
import meshio
import os
import sys
import yaml
from colorama import Fore, Style
from results import test_graphs

# import from utils that is in the same directory as this file
# change the path to the directory of this file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import analytical

# Test parameters
mesh_dir    = "tests/utils/altered_mesh/"
output_dir  = "tests/utils/result_mesh/"
n_files     = -1

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def l2norm_relative(measure, reference):
    sqr_sum = np.sum(reference ** 2)
    return np.sqrt(np.sum((measure - reference) ** 2) / sqr_sum)

def l2norm_array(measure, reference):
    sqr_sum = np.sum(reference ** 2)
    return np.sqrt(((measure - reference) ** 2) / sqr_sum) 

FUNCTIONS = [analytical.linear, analytical.quadratic, analytical.quarter_five_spot]
NAMES     = ["linear", "quadratic", "quarter_five_spot"]

class TestAccuracy:
    
    def test_accuracy(self):
        """
        Tests whether the interpolator is accurate.
        """
        global n_files
        # Iterate over n_files files of mesh_dir
        files   = sorted(os.listdir(mesh_dir))
        # Remove .gitkeep
        if ".gitkeep" in files:
            files.remove(".gitkeep")
        
        if n_files == -1:
            n_files = len(files)
        
        results_dict = {}
        results_dict["n_files"] = n_files
        results_dict["files"] = {}

        print("\n=========================================================")
        print(f"{Fore.WHITE}{'File':<15}{'Variable':<20}{'Method':<10}{'Error'}{Style.RESET_ALL}")
        for case in range(n_files):
            print(f"{Fore.BLUE}{files[case]:<15}{f'{case}/{n_files}':<10}{Style.RESET_ALL}")

            interpolador = ninpol.Interpolator()
            interpolador.load_mesh(mesh_dir + files[case])
            
            points_coords = np.asarray(interpolador.grid_obj.point_coords)
            n_points = points_coords.shape[0]
            points = np.arange(n_points)

            results_dict["files"][files[case]] = {}
            results_dict["files"][files[case]]["n_points"] = n_points
            results_dict["files"][files[case]]["methods"] = {}
            msh = meshio.read(mesh_dir + files[case])
            point_data = msh.point_data

            for method in interpolador.supported_methods.keys():
                results_dict["files"][files[case]]["methods"][method] = {}
                for function, name in zip(FUNCTIONS, NAMES):
                    
                    if name not in msh.cell_data:
                        continue
                        
                    reference = function(points_coords)
                    measure = np.asarray(interpolador.interpolate(name, method, return_value=True))
                    norm = float(l2norm_relative(measure, reference))
                    
                    results_dict["files"][files[case]]["methods"][method]["error_" + name] = norm
                    print(f"{Fore.LIGHTWHITE_EX}{'  ~  ':<15}{name:<20}{Fore.LIGHTYELLOW_EX}{method:<10}{Fore.LIGHTRED_EX}{norm:2e}{Style.RESET_ALL}")
                    point_data[name + "_" + method] = measure
                    point_data[name + "_" + method + "_error"] = l2norm_array(measure, reference)

            mesh_out = meshio.Mesh(msh.points, msh.cells, cell_data=msh.cell_data, point_data=point_data)
            # Remove extension from file name
            filename_without_extension = os.path.splitext(files[case])[0]
            block_print()
            meshio.write(output_dir + filename_without_extension + ".vtu", mesh_out)
            enable_print()
            print("=========================================================")
        
        import datetime
        results_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open("tests/results/accuracy_test.yaml", "w") as f:
            yaml.dump(results_dict, f)

        test_graphs.graph_accuracy()
