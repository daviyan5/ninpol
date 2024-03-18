import pytest
import numpy as np
import ninpol
import meshio
import os
import sys
import yaml
from colorama import Fore, Style

# import from utils that is in the same directory as this file
# change the path to the directory of this file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import analytical

# Test parameters
mesh_dir    = "tests/utils/altered/"
output_dir  = "tests/utils/results/"

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def l2norm_relative(measure, reference):
    sqr_sum = np.sum(reference ** 2)
    return np.sqrt(np.sum((measure - reference) ** 2) / sqr_sum)

FUNCTIONS = [analytical.linear, analytical.quadratic]
NAMES     = ["linear", "quadratic"]

class TestInterpolator:
    def test_accuracy(self):
        """
        Tests whether the interpolator is accurate.
        """
        # Iterate over n_files files of mesh_dir
        files   = sorted(os.listdir(mesh_dir))
        n_files = len(files)
        
        results = {}
        results["n_files"] = n_files
        results["files"] = {}

        print()
        print(f"{Fore.WHITE}{'File':<10}{'Variable':<20}{'Method':<10}{'Error'}{Style.RESET_ALL}")
        for case in range(n_files):
            print(f"{Fore.BLUE}{files[case]:<10}{f'{case}/{n_files}':<10}{Style.RESET_ALL}")

            interpolador = ninpol.Interpolator()
            interpolador.load_mesh(mesh_dir + files[case])
            
            points_coords = np.asarray(interpolador.grid_obj.point_coords)
            n_points = points_coords.shape[0]
            points = np.arange(n_points)

            results["files"][files[case]] = {}
            msh = meshio.read(mesh_dir + files[case])
            point_data = msh.point_data

            for method in interpolador.supported_methods.keys():
                results["files"][files[case]][method] = {}
                for function, name in zip(FUNCTIONS, NAMES):
                    reference = function(points_coords)
                    measure = np.asarray(interpolador.interpolate(points, name, method, "ctp"))
                    norm = float(l2norm_relative(measure, reference))
                    results["files"][files[case]][method]["error_" + name] = norm
                    print(f"{Fore.LIGHTWHITE_EX}{'  -  ':<10}{name:<20}{Fore.LIGHTYELLOW_EX}{method:<10}{Fore.LIGHTRED_EX}{norm:2e}{Style.RESET_ALL}")
                    point_data[name + "_" + method] = measure
            
            mesh_out = meshio.Mesh(msh.points, msh.cells, cell_data=msh.cell_data, point_data=point_data)
            # Remove extension from file name
            filename_without_extension = os.path.splitext(files[case])[0]
            blockPrint()
            meshio.write(output_dir + filename_without_extension + ".vtu", mesh_out)
            enablePrint()

        with open("tests/results/accuracy_test.yaml", "w") as f:
            yaml.dump(results, f)

