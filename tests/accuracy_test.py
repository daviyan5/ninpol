import pytest
import numpy as np
import multiprocessing as mp

import psutil
import subprocess

import ninpol
import time
import meshio
import pickle
import datetime
import os
import sys
import yaml
import gc

# import from utils that is in the same directory as this file
# change the path to the directory of this file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.analytical import LINCase, QUADCase, FANcase, ALHcase

from colorama import Fore, Style
from results import test_graphs
from memory_profiler import memory_usage

# Test parameters
mesh_dir    = "./mesh"
output_dir  = "./results/mesh/"

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def write_results(results_dict):
    with open("./results/accuracy.yaml", "w") as f:
        yaml.dump(results_dict, f)

def export_vtu(cells, points, cell_data, point_data, output_dir, filename):
    mesh = meshio.Mesh(points, cells, cell_data=cell_data, point_data=point_data)
    mesh_filename = f"{output_dir}/{filename}_acc.vtu"

    if os.path.exists(mesh_filename):
        meshio.write(mesh_filename, mesh)
    
    return mesh_filename

class TestAccuracy:
    
    def test_accuracy(self):
        """
        Tests whether the interpolator is accurate.
        """
        # cd to the directory of this file
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        files   = sorted(os.listdir(mesh_dir))
        # Remove .gitkeep
        if ".gitkeep" in files:
            files.remove(".gitkeep")
        
        max_size = 100
        # Remove "box" meshes
        new_files = []
        for file in files:
            # search for "box" in the filename
            # Also eliminate files > max_size
            if "box" not in file and "0" not in file:
                if os.path.getsize(os.path.join(mesh_dir, file)) < max_size * 1024 * 1024:
                    new_files.append(file)
        files = new_files

        # sort files by size
        files = sorted(files, key=lambda x: os.path.getsize(os.path.join(mesh_dir, x)))
        n_files = len(files)
        
        cases = [
            LINCase(), QUADCase(), FANcase(), ALHcase()
        ]

        n_cases = len(cases)

        def style(index, file, n_points, case, method, error, header=None):
            if header is not None:
                index_s    = f"[{index.upper():<8}]"
                file_s     = f"{file.upper():<10}"
                n_points_s = f"{n_points.upper():<7}"
                case_s     = f"{case.upper():<5}"
                method_s   = f"{method.upper():<6}"
                error_s    = f"{error.upper():<5}"
            else:
                index_s    = f"{Fore.WHITE}[{index:<8}]{Style.RESET_ALL}"
                file_s     = f"{Fore.LIGHTCYAN_EX}{file:<10}{Style.RESET_ALL}"
                n_points_s = f"{Fore.LIGHTYELLOW_EX}{n_points:<7}{Style.RESET_ALL}"
                case_s     = f"{Fore.LIGHTBLUE_EX}{case:<5}{Style.RESET_ALL}"
                method_s   = f"{Fore.LIGHTGREEN_EX}{method:<6}{Style.RESET_ALL}"
                error_s    = f"{Fore.LIGHTMAGENTA_EX}{error:<5.3e}{Style.RESET_ALL}"

            print(f"{index_s} {file_s} | {n_points_s} | {case_s} | {method_s} | {error_s}")

        print("\n============================================================")        
        style("INDEX", "FILE", "POINT", "CASE", "METHOD", "ERROR", header=True)

        mesh_types = ["hexa", "tetra", "prism"]
        interpolator = ninpol.Interpolator(logging=False, build_edges=False)

        results_dict = {
            case.name: {
                mtype : {
                    method: {
                        "error": [],
                    } for method in interpolator.supported_methods
                } for mtype in mesh_types
            } for case in cases
        }
        for case in cases:
            for mtype in mesh_types:
                results_dict[case.name][mtype]["n_points"] = []
                results_dict[case.name][mtype]["n_vols"]   = []

        results_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        for i, mesh_filename in enumerate(files):
            
            cells       = {}
            points      = []
            cell_data   = {}
            point_data  = {}

            for j, case in enumerate(cases):
                mesh_path = os.path.join(mesh_dir, mesh_filename)
                mtype = [mtype for mtype in mesh_types if mtype in mesh_filename][0]

                block_print()
                if os.path.exists(f"/tmp/{case.name}_{mesh_filename}.pkl"):
                    case = pickle.load(open(f"/tmp/{case.name}_{mesh_filename}.pkl", "rb"))
                else:
                    case.assign_mesh_properties(mesh_path)
                    pickle.dump(case, open(f"/tmp/{case.name}_{mesh_filename}.pkl", "wb"))
                enable_print()

                if j == 0:
                    cells       = case.mesh.cells
                    points      = case.mesh.points
                
                for key in case.mesh.cell_data:
                    cell_data[key] = case.mesh.cell_data[key]
                
                for key in case.mesh.point_data:
                    point_data[key] = case.mesh.point_data[key]
                
                n_methods = len(interpolator.supported_methods)

                print(f"Building {mesh_filename}...", end="\r")
                block_print()
                interpolator.load_mesh(mesh_obj=case.mesh)
                enable_print()
                print("" * 200, end="\r")

                for k, method in enumerate(interpolator.supported_methods):

                    print(f"Running {case.name} on {mesh_filename} with {method}...", end="\r")

                    weights, __ = interpolator.interpolate(case.name, method)
                        
                    print("" * 200, end="\r")

                    error, error_arr = case.evaluate(weights)

                    point_data[f"l2_{case.name}_{method}"] = error_arr
                    index = i * n_cases * n_methods + j * n_methods + k
                    total = n_files * n_cases * n_methods
                    idx_str = str(index + 1) + '/' + str(total)
                    style(idx_str, mesh_filename, case.mesh.points.shape[0], case.name, method, error)

                    
                    results_dict[case.name][mtype][method]["error"].append(float(error))
                    write_results(results_dict)              
                
                results_dict[case.name][mtype]["n_points"].append(case.mesh.points.shape[0])
                results_dict[case.name][mtype]["n_vols"].append(case.mesh.cells[0].data.shape[0])

            vtu_filename = export_vtu(cells, points, cell_data, point_data, output_dir, mesh_filename.split(".")[0])
            print(f"Exported {vtu_filename}...")
            print("============================================================")
            if i < n_files - 1:
                style("INDEX", "FILE", "POINT", "CASE", "METHOD", "ERROR", header=True)