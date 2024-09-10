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
    with open("./results/performance.yaml", "w") as f:
        yaml.dump(results_dict, f)

def export_vtk(mesh, output_dir, filename):
    cells  = mesh.cells
    points = mesh.points
    cell_data  = mesh.cell_data
    point_data = mesh.point_data
    mesh_filename = f"{output_dir}/{filename}.vtk"
    if os.path.exists(mesh_filename):
        return mesh_filename
    meshio.write(mesh_filename, meshio.Mesh(points, cells, cell_data=cell_data, point_data=point_data))
    return mesh_filename

def run_script_in_process(script_path):
    # Function to execute the Python script using subprocess and return the process PID
    process = subprocess.Popen(['python3', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def monitor_memory(pid):
    # Monitor memory usage of the process by its PID using psutil
    try:
        proc = psutil.Process(pid)
        mem_usage = []
        
        # Monitor memory usage while the process is running
        while proc.is_running():
            try:
                # Check if the process is still alive and not a zombie
                if proc.status() != psutil.STATUS_ZOMBIE:
                    mem_info = proc.memory_info().rss / (1024 ** 2)  # Memory in MB
                    mem_usage.append(mem_info)
                else:
                    break
            except psutil.NoSuchProcess:
                # If the process no longer exists, break the loop
                break
            time.sleep(0.1)  # Sleep for 100 ms before the next memory check

        return mem_usage
    except psutil.NoSuchProcess:
        return []

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
        
        max_size = 2
        # Remove "box" meshes
        new_files = []
        for file in files:
            # search for "box" in the filename
            # Also eliminate files > max_size
            if "box" not in file and "0" not in file:
                if os.path.getsize(os.path.join(mesh_dir, file)) < max_size * 1024 * 1024:
                    new_files.append(file)
        files = new_files
        n_files = len(files)
        
        cases = [
            LINCase(), QUADCase(), FANcase(), ALHcase()
        ]

        n_cases = len(cases)

        def style(index, file, case, method, build_time, interpolate_time, memory):

            index_s             = f"{Fore.WHITE}[{index:<8}]{Style.RESET_ALL}"
            file_s              = f"{Fore.LIGHTCYAN_EX}{file:<10}{Style.RESET_ALL}"
            case_s              = f"{Fore.LIGHTBLUE_EX}{case:<5}{Style.RESET_ALL}"
            method_s            = f"{Fore.LIGHTGREEN_EX}{method:<5}{Style.RESET_ALL}"
            build_time_s        = f"{Fore.LIGHTMAGENTA_EX}{build_time:<5.3f}s{Style.RESET_ALL}"
            interpolate_time_s  = f"{Fore.LIGHTYELLOW_EX}{interpolate_time:<5.3f}s{Style.RESET_ALL}"
            total_time_s        = f"{Fore.LIGHTRED_EX}{build_time + interpolate_time:<5.3f}s{Style.RESET_ALL}"
            memory_s            = f"{Fore.LIGHTWHITE_EX}{memory:.3f} MB{Style.RESET_ALL}"

            print(f"{index_s} {file_s} | {case_s} | {method_s} | {build_time_s} | {interpolate_time_s} | {total_time_s} | {memory_s}")

        print("\n===========================================================================")        

        mesh_types = ["hexa", "tetra", "prism"]
        interpolator = ninpol.Interpolator(logging=False, build_edges=False)

        results_dict = {
            case.name: {
                mtype : {
                    method: {
                        "time": [],
                        "memory": []
                    } for method in interpolator.supported_methods
                } for mtype in mesh_types
            } for case in cases
        }
        results_dict["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        for case in cases:
            for mtype in mesh_types:
                results_dict[case.name][mtype]["build"] = []

        for i, mesh_filename in enumerate(files):
            
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

                
                n_methods = len(interpolator.supported_methods)
                n_repeats = 3

                build_time = 0.
                for r in range(n_repeats):
                    print(f"Building {mesh_filename} ({r+1}/{n_repeats})", end="\r")
                    block_print()
                    build_time -= time.time()
                    interpolator.load_mesh(mesh_obj=case.mesh)
                    build_time += time.time()
                    enable_print()

                

                # Create a process to run the script and use memory_profiler to get the memory usage
                build_time /= n_repeats
                results_dict[case.name][mtype]["build"].append(build_time)
                print("" * 200, end="\r")

                for k, method in enumerate(interpolator.supported_methods):
                    interpolate_time = 0.
                    for r in range(n_repeats):
                        print(f"Running {case.name} on {mesh_filename} with {method} ({r+1}/{n_repeats})", end="\r")

                        interpolate_time -= time.time()
                        _, __ = interpolator.interpolate(case.name, method)
                        interpolate_time += time.time()
                        
                    print("" * 200, end="\r")

                    vtk_filename = export_vtk(case.mesh, "/tmp", mesh_filename.split(".")[0])

                    create_script("/tmp/script.py", case.name, method, vtk_filename)
                    
                    
                    # Create a process to run the script and use memory_profiler to get the memory usage
                    process = run_script_in_process("/tmp/script.py")
                    pid = process.pid
                    memory_usage_list = monitor_memory(pid)
                    
                    process.wait()

                    memory = max(memory_usage_list) if memory_usage_list else 0.0

                    interpolate_time /= n_repeats
                    
                    index = i * n_cases * n_methods + j * n_methods + k
                    total = n_files * n_cases * n_methods
                    idx_str = str(index + 1) + '/' + str(total)
                    style(idx_str, mesh_filename, case.name, method, build_time, interpolate_time, memory)

                   
                    results_dict[case.name][mtype][method]["time"].append(interpolate_time)
                    results_dict[case.name][mtype][method]["memory"].append(memory)
                    write_results(results_dict)              

            print("===========================================================================")

def create_script(script_path, case_name, method, vtk_filename):
    # Write the Python script to the specified path
    script = f"""
import ninpol
import time

time.sleep(1)
interpolator = ninpol.Interpolator()
interpolator.load_mesh('{vtk_filename}')
interpolator.interpolate('{case_name}', '{method}')
"""
    with open(script_path, "w") as f:
        f.write(script)
