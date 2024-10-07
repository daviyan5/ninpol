
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
    for i in range(1, len(errors)):
        ratio = (np.log(errors[i] / errors[i - 1])) / (np.log(n_points[i] / n_points[i - 1]))
        Ru.append(float(-3 * ratio))
    Ru.insert(0, None)
    return Ru

# Define a dictionary for consistent colors across methods
colors = {'gls': 'blue', 'idw': 'orange', 'ls': 'green'}
linestyles = {'gls': '-', 'idw': '--', 'ls': '-.'}
mcolors = {'hexa': 'red', 'tetra': 'blue', 'prism': 'green'}

def build_graph(build_times, n_points, mesh_types):
    fig, ax = plt.subplots()

    markers = ['o', 's', 'D']  # Different marker styles for different mesh types
    for i, mtype in enumerate(mesh_types):
        ax.plot(n_points[mtype], build_times[mtype], 
                label=mtype, color=mcolors[mtype], marker=markers[i % len(markers)], 
                linewidth=2.0, markersize=5)
        # Plot, in traced lines ---, a linear interpolation of build_time for n_points = 0 and n_points = 1e6
        start_p = 0
        end_p   = 2.1e6

        a = (build_times[mtype][-1] - build_times[mtype][0]) / (n_points[mtype][-1] - n_points[mtype][0])
        
        b = build_times[mtype][0] - a * n_points[mtype][0]

        ax.plot([start_p, end_p], [a * start_p + b, a * end_p + b],
                color=mcolors[mtype], linestyle='--', linewidth=2.0)

    ax.set_xlabel("Número de Pontos")
    ax.set_ylabel("Tempo de Montagem (s)")
    ax.set_title("Tempo de Montagem para Diferentes Tipos de Malha", fontsize=14)

    ax.minorticks_on()
    ax.grid(which='both', linestyle='--', linewidth=0.5)

    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize='large')

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_folder, "build_times.png"), dpi=300)

def performance_graph_multi(case, n_points, times_by_method, memory_by_method, mesh_types):
    # Create figure for execution time (3 subplots, one for each mesh type)
    fig_time, ax_time = plt.subplots(1, 3, figsize=(18, 6))
    fig_time.suptitle(f"Tempo de Execução para {case}", fontsize=16)

    # Create figure for memory (3 subplots, one for each mesh type)
    fig_memory, ax_memory = plt.subplots(1, 3, figsize=(18, 6))
    fig_memory.suptitle(f"Memória Máxima Alocada para {case}", fontsize=16)

    markers = ['o', 's', 'D']  # Different marker styles for different methods

    # Loop over each mesh type to create separate subplots
    for i, mtype in enumerate(mesh_types):
        ax_time[i].set_title(f"Malha: {mtype}", fontsize=14)
        ax_memory[i].set_title(f"Malha: {mtype}", fontsize=14)

        # Plot execution time with markers on the corresponding subplot
        for j, (method, times) in enumerate(times_by_method[mtype].items()):
            ax_time[i].plot(n_points[mtype], times, 
                            label=method, color=colors[method], marker=markers[j % len(markers)], 
                            linestyle=linestyles[method],
                            linewidth=2.0, markersize=8)
        ax_time[i].set_xlabel("Número de Pontos")
        ax_time[i].set_ylabel("Tempo de Execução (s)")
        ax_time[i].set_yscale("log")
        ax_time[i].set_xscale("log")
        ax_time[i].minorticks_on()
        ax_time[i].grid(which='both', linestyle='--', linewidth=0.5)

        # Plot memory performance with markers on the corresponding subplot
        for j, (method, memory) in enumerate(memory_by_method[mtype].items()):
            ax_memory[i].plot(n_points[mtype], memory, 
                              label=method, color=colors[method], 
                              linestyle=linestyles[method],
                              marker=markers[j % len(markers)], 
                              linewidth=2.0, markersize=8)
        ax_memory[i].set_xlabel("Número de Pontos")
        ax_memory[i].set_ylabel("Memória Alocada (MB)")
        ax_memory[i].minorticks_on()
        ax_memory[i].grid(which='both', linestyle='--', linewidth=0.5)

        # Add legends to each subplot
        ax_time[i].legend(loc='upper left', frameon=True, shadow=True, fontsize='large')
        ax_memory[i].legend(loc='upper left', frameon=True, shadow=True, fontsize='large')

    # Adjust layout for both figures
    fig_time.tight_layout()
    fig_memory.tight_layout()
    plt.tight_layout()

    # Save the figures to disk
    fig_time.savefig(os.path.join(graphs_folder, f"time_{case}.png"), dpi=300)
    fig_memory.savefig(os.path.join(graphs_folder, f"memory_{case}.png"), dpi=300)


def accuracy_csv(case, mtype, n_points, accuracy_by_method, methods):
    with open(f"{csv_folder}/{case}_{mtype}.csv", "w") as file:
        writer = csv.writer(file)
        header = ["case_name", "n_points"]
        for method in methods:
            header += [f"{method}_error", f"{method}_Ru"]
        writer.writerow(header)
        for i in range(len(n_points[mtype])):
            row = [case, n_points[mtype][i]]
            for method in methods:
                row.append(np.round(accuracy_by_method[method]["l2"][i], 2))
                Ru = accuracy_by_method[method]["Ru"][i] if accuracy_by_method[method]["Ru"][i] is not None else "--"
                if Ru != "--":
                    Ru = np.round(Ru, 2)
                row.append(Ru)
            writer.writerow(row)

def mpfa_csv(case, mpfa_n_points, mpfa_ninpol_times, mpfa_py_times, accuracy_per_method, methods):
    with open(f"{csv_folder}/mpfa_{case}.csv", "w") as file:

        writer = csv.writer(file)
        row = [case]
        for method in mpfa_ninpol_times:
            row.append(method + "_speedup")
            row.append(method + "_Ru")
        writer.writerow(row)
        last_error = {method: None for method in mpfa_ninpol_times}

        for i in range(len(mpfa_n_points)):
            row = [mpfa_n_points[i]]
            for method1, method2 in zip(mpfa_ninpol_times, mpfa_py_times):
                speedup = calc_speedup(mpfa_ninpol_times[method1], mpfa_py_times[method2])
                row.append(np.round(speedup[i], 2))
                if last_error[method1] == None:
                    row.append("--")
                    last_error[method1] = accuracy_per_method[method1][i]
                else:
                    Ru = calc_Ru([last_error[method1], accuracy_per_method[method1][i]], [mpfa_n_points[i - 1], mpfa_n_points[i]])[1]
                    row.append(np.round(Ru, 2))
                    last_error[method1] = accuracy_per_method[method1][i]
            writer.writerow(row)
        
def plot_accuracy_multi(case, n_points, accuracy_by_method, methods, mesh_types):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Acurácia (Norma L2) para {case}", fontsize=16)

    markers = ['o', 's', 'D']  # Different marker styles for different methods

    # Loop over each mesh type to create separate subplots
    for i, mtype in enumerate(mesh_types):
        ax[i].set_title(f"Malha: {mtype}", fontsize=14)

        # Plot L2 error performance with markers
        minn  = min(n_points[mtype])
        minerr = min([min(accuracy_by_method[mtype][method]["l2"]) for method in methods])
        if case != "LIN":
            ax[i].plot([minn, 10 * minn], [minerr, minerr * np.exp((-2 / 3) * np.log(10))], 
                        color='r', label="O(n²)")

        for j, (method, errors) in enumerate(accuracy_by_method[mtype].items()):
            error = np.array(errors["l2"])
            error[error == 0] = 1e-16 
            ax[i].plot(n_points[mtype], error, 
                       label=method, color=colors[method], 
                       linestyle=linestyles[method],
                       marker=markers[j % len(markers)], 
                       linewidth=2.0, markersize=8)
        
        # If case == "LIN", set the max and min of the y-axis to 0 and 1
        
            
        
        ax[i].set_xlabel("Número de Pontos")
        ax[i].set_ylabel("Norma L2")
        ax[i].set_yscale("log")
        ax[i].set_xscale("log")
        ax[i].minorticks_on()
        ax[i].grid(which='both', linestyle='--', linewidth=0.5)
        if case == "LIN":
            ax[i].set_ylim(1e-18, 100)
        # Add legends to each subplot
        ax[i].legend(loc='upper left', frameon=True, shadow=True, fontsize='large')

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_folder, f"acc_{case}.png"), dpi=300)

def graph():
    # ----- Build graph -----
    cases      = list(performance_data.keys())
    mesh_types = list(performance_data[cases[0]].keys())
    methods    = list(performance_data[cases[0]][mesh_types[0]]["methods"].keys())

    build_times = {mtype: performance_data[cases[0]][mtype]["build"] for mtype in mesh_types}
    n_points    = {mtype: performance_data[cases[0]][mtype]["n_points"] for mtype in mesh_types}
    build_graph(build_times, n_points, mesh_types)

    for case in cases:
        times_by_method = {mtype: {method: performance_data[case][mtype]["methods"][method]["time"] for method in methods} for mtype in mesh_types}
        memory_by_method = {mtype: {method: performance_data[case][mtype]["methods"][method]["memory"] for method in methods} for mtype in mesh_types}
        performance_graph_multi(case, n_points, times_by_method, memory_by_method, mesh_types)

        if case != "LIN":
            mtype = "tetra"
            accuracy_by_method = {method: {"l2": accuracy_data[case][mtype]["methods"][method]["error"] } for method in methods}
            for method in methods:
                accuracy_by_method[method]["Ru"] = calc_Ru(accuracy_by_method[method]["l2"], n_points[mtype])

            accuracy_csv(case, mtype, n_points, accuracy_by_method, methods)

        accuracy_by_method = {mtype: {method: {"l2": accuracy_data[case][mtype]["methods"][method]["error"] } for method in methods} for mtype in mesh_types}
        plot_accuracy_multi(case, n_points, accuracy_by_method, methods, mesh_types)

    mpfa_cases = list(mpfa_data.keys())
    mpfa_methods = [method for method in mpfa_data[mpfa_cases[0]]]
    mpfa_n_points = {case: mpfa_data[case][mpfa_methods[0]]["n_points"] for case in mpfa_cases}
    mpfa_ninpol_times = {case: {method: mpfa_data[case][method]["times"] for method in mpfa_methods if "ninpol" in method} for case in mpfa_cases}
    mpfa_py_times = {case: {method: mpfa_data[case][method]["times"] for method in mpfa_methods if "py" in method} for case in mpfa_cases}
    mpfa_accuracy_per_method = {case: {method: mpfa_data[case][method]["accuracy"] for method in mpfa_methods if "ninpol" in method} for case in mpfa_cases}
    for case in mpfa_cases:
        mpfa_csv(case, mpfa_n_points[case], mpfa_ninpol_times[case], mpfa_py_times[case], mpfa_accuracy_per_method[case], mpfa_methods)

if __name__ == "__main__":
    graph()
