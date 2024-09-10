
"""
N = n_points
First, output a single graph that shows the time taken x N for each mesh type to build the mesh. (build_times.png)
Then, output, for each case:
    For each mtype:
        1x2 Figure, with time on the left and memory on the right. ({case_name}_{mtype}.png)
            - Time x N graph for each method on the same figure  (log scale on both axes)
            - Memory x N graph for each method on the same figure 

        Accuracy csv: ({case.name}_{mesh_type}.csv):
            |                       case.name                       |
            |          | Method 1   | Method 2   | ... | Method N   | 
            |n_points  | error | Ru | error | Ru | ... | error | Ru |
            |...       | ...   | ...| ...   | ...| ... | ...   | ...|

    Output a table (mpfa.csv):
        |                           case.name                      |
        |           | Method 1  | Method 2   | ...    | Method N   |
        | n_points  | speedup   | speedup    | ...    | speedup    |

"""
# performance.yaml structure:
"""
case.name: {
    mtype: {
        "build": [],
        "n_points": [],
        "methods": {
            method: {
                "time": [],
                "memory: []
            } for method in interpolator.supported_methods
        }
    } for mtype in mtypes
} for case in cases

"""

# accuracy.yaml structure:
"""
case.name: {
    mtype: {
        method: {
            "error": []
        } for method in interpolator.supported_methods
    } for mtype in mtypes
} for case in cases
"""

# mpfa.yaml structure:
"""
case.name: {
    method: {
        "time": [],
        "accuracy": [],
        "n_points": []
    }    
}
"""

# Steps:
# 1. Load performance.yaml, accuracy.yaml, and mpfa.yaml
# 2. Calculate speedup for each method in each case

"""
speedup = method_2_time / method_1_time

Inside the mpfa.yaml file, we have:
method_1 = "ninpol_(method_name)"
method_2 = "py_(method_name)"

"""
# 3. Calculate Ru for each method in each case

"""
Ru(e_i, e_i+1, N_i, N_i+1) = -3 * (log(e(i+1)) / log(e(i)))
                                    (log(N(i+1)) - log(N(i)))
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml
import csv
import os

yaml_folder   = "yaml"
graphs_folder = "graphs"
csv_folder    = "csv"

# CD to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load YAML files
with open(os.path.join(yaml_folder, './performance.yaml'), 'r') as file:
    performance_data = yaml.safe_load(file)

with open(os.path.join(yaml_folder,'./accuracy.yaml'), 'r') as file:
    accuracy_data = yaml.safe_load(file)

with open(os.path.join(yaml_folder,'./mpfa.yaml'), 'r') as file:
    mpfa_data = yaml.safe_load(file)

def calc_speedup(time_1, time_2):
    return np.array(time_2) / np.array(time_1)

def calc_Ru(errors, n_points):
    Ru = []
    for i in range(len(errors) - 1):
        ratio = (np.log(errors[i + 1]) - np.log(errors[i])) / (np.log(n_points[i + 1]) - np.log(n_points[i]))
        Ru.append(-3 * ratio)
    Ru.insert(0, None)
    return Ru

def build_graph(build_times, n_points, mesh_types):
    fig, ax = plt.subplots()
    for i, mtype in enumerate(mesh_types):
        ax.plot(n_points[mtype], build_times[mtype], label=mtype)
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Time (s)")

    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_folder, "./build_times.png"))

def performance_graph(case, mtype, n_points, times_by_method, memory_by_method):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title(f"{case} - {mtype}")
    ax[1].set_title(f"{case} - {mtype}")
    for method, times in times_by_method.items():
        ax[0].plot(n_points[mtype], times, label=method)
    ax[0].set_xlabel("Number of points")
    ax[0].set_ylabel("Time (s)")
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    

    for method, memory in memory_by_method.items():
        ax[1].plot(n_points[mtype], memory, label=method)
    ax[1].set_xlabel("Number of points")
    ax[1].set_ylabel("Memory (MB)")

    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_folder, f"./{case}_{mtype}.png"))

def accuracy_csv(case, mtype, n_points, accuracy_by_method, methods):
    with open(f"{csv_folder}/{case}_{mtype}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["case_name", "n_points"] + [f"{method}_error" for method in methods] + [f"{method}_Ru" for method in methods])
        for i in range(len(n_points[mtype])):
            row = [case, n_points[mtype][i]]
            for method in methods:
                row.append(accuracy_by_method[method]["l2"][i])
                row.append(accuracy_by_method[method]["Ru"][i])
            writer.writerow(row)

def mpfa_csv(case, mpfa_n_points, mpfa_ninpol_times, mpfa_py_times, methods):
    with open(f"{csv_folder}/mpfa_{case}.csv", "w") as file:

        writer = csv.writer(file)
        writer.writerow([case] + [f"{method}_speedup" for method in methods])
        
        for i in range(len(mpfa_n_points)):
            row = [mpfa_n_points[i]]
            for method1, method2 in zip(mpfa_ninpol_times, mpfa_py_times):
                speedup = calc_speedup(mpfa_ninpol_times[method1], mpfa_py_times[method2])
                row.append(speedup[i])
            writer.writerow(row)
        

def graph():
    # ----- Build graph -----
    cases      = list(performance_data.keys())
    mesh_types = list(performance_data[cases[0]].keys())
    methods    = list(performance_data[cases[0]][mesh_types[0]]["methods"].keys())

    build_times = {mtype: performance_data[cases[0]][mtype]["build"] for mtype in mesh_types}
    n_points    = {mtype: performance_data[cases[0]][mtype]["n_points"] for mtype in mesh_types}
    build_graph(build_times, n_points, mesh_types)

    for case in cases:
        for mtype in mesh_types:
            # ----- Time x N / Memory x N graphs -----
            if n_points[mtype] == []:
                continue
            times_by_method  = {method: performance_data[case][mtype]["methods"][method]["time"] for method in methods}
            memory_by_method = {method: performance_data[case][mtype]["methods"][method]["memory"] for method in methods}
            performance_graph(case, mtype, n_points, times_by_method, memory_by_method)
            
        if case != "LIN":
            mtype = mesh_types[0]
            accuracy_by_method = {method: {"l2": accuracy_data[case][mtype]["methods"][method]["error"] } for method in methods}
            for method in methods:
                accuracy_by_method[method]["Ru"] = calc_Ru(accuracy_by_method[method]["l2"], n_points[mtype])
        
            accuracy_csv(case, mtype, n_points, accuracy_by_method, methods)

    mpfa_cases    = list(mpfa_data.keys())
    mpfa_methods  = [method for method in mpfa_data[mpfa_cases[0]]]
    mpfa_n_points = {case: mpfa_data[case][mpfa_methods[0]]["n_points"] for case in mpfa_cases}
    mpfa_ninpol_times = {case: {method: mpfa_data[case][method]["times"] for method in mpfa_methods if "ninpol" in method} for case in mpfa_cases}
    mpfa_py_times     = {case: {method: mpfa_data[case][method]["times"] for method in mpfa_methods if "py" in method} for case in mpfa_cases}
    
    for case in mpfa_cases:
        mpfa_csv(case, mpfa_n_points[case], mpfa_ninpol_times[case], mpfa_py_times[case], mpfa_methods)



if __name__ == "__main__":
    graph()
