import os
import sys
import meshio
import numpy as np
import analytical

import ninpol
# Move current path to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def calculate_properties(centroids, is_box=False):
    # Calculate properties for each centroid using the analytical package
    linear              = analytical.linear(centroids)
    quadratic           = analytical.quadratic(centroids)
    quarter_five_spot   = analytical.quarter_five_spot(centroids, is_box)

    return linear, quadratic, quarter_five_spot

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def process_mesh_file(file_name, file_path, output_dir, temp_output_dir):

    print("Processing file:", file_name)

    block_print()
    # Read mesh file using meshio
    mesh = meshio.read(file_path)
    # Write in a temporary output directory
    meshio.write(temp_output_dir + "foo.vtk", mesh)
    enable_print()

    # Read mesh file using meshio
    mesh = meshio.read(temp_output_dir + "foo.vtk")
    
    mesh.points = mesh.points.astype(float)

    n_cells = 0
    connectivity = None
    linear = []
    quadratic = []
    permeability = []
    quarter_five_spot = []
    for cell_type in mesh.cells:
        n_cells = len(cell_type.data)
        cell_type.data = cell_type.data.astype(int)
        connectivity = cell_type.data

        centroids = np.mean(mesh.points[connectivity], axis=1)
        lp, q, q5 = calculate_properties(centroids, "box" in file_name)
        linear.append(lp)
        quadratic.append(q)
        K = np.ndarray((n_cells, 3 * 3))
        K[:] = np.array([1.0, 0.5, 0,
                         0.5, 1.0, 0.5,
                           0, 0.5, 1.0])
        
        permeability.append(K)
        if "box" in file_name:
            quarter_five_spot.append(q5)

    interpolador = ninpol.Interpolator()
    interpolador.load_mesh(temp_output_dir + "foo.vtk")
    grid_obj = interpolador.grid_obj
    
    neumann_flag = np.zeros(grid_obj.n_points, dtype=int)
    dirichlet_flag = np.zeros(grid_obj.n_points, dtype=int)

    neumann_l      = np.zeros(grid_obj.n_points)
    dirichlet_l      = np.zeros(grid_obj.n_points)

    neumann_q      = np.zeros(grid_obj.n_points)
    dirichlet_q      = np.zeros(grid_obj.n_points)

    neumann_q5      = np.zeros(grid_obj.n_points)
    dirichlet_q5      = np.zeros(grid_obj.n_points)
    
    for face in range(grid_obj.n_faces):
        if grid_obj.boundary_faces[face]:
            psuf = np.asarray(grid_obj.inpofa[face])
            psuf = psuf[psuf != -1]
            rd = 1.2# p.random.rand()
            if rd < (0.3 if not "box" in file_name else 1.0):
                neumann_flag[psuf] = True
                neumann_l[psuf] = analytical.get_bc(grid_obj.faces_centers[face], permeability[0][0].reshape(3, 3),
                                                                           "linear", "neumann")
                
                neumann_q[psuf] = analytical.get_bc(grid_obj.faces_centers[face], permeability[0][0].reshape(3, 3),
                                                                           "quadratic", "neumann", normal=grid_obj.normal_faces[face])
                
                if "box" in file_name:
                    neumann_q5[psuf] = analytical.get_bc(grid_obj.faces_centers[face], permeability[0][0].reshape(3, 3),
                                                                           "quarter_five_spot", "neumann")
            else:
                psuf = psuf[neumann_flag[psuf] == 0]
                dirichlet_flag[psuf] = True
                
                dirichlet_l[psuf] = analytical.get_bc(grid_obj.faces_centers[face], permeability[0][0].reshape(3, 3),
                                                                            "linear", "dirichlet")

                dirichlet_q[psuf] = analytical.get_bc(grid_obj.faces_centers[face], permeability[0][0].reshape(3, 3),
                                                                            "quadratic", "dirichlet")
                
                if "box" in file_name:
                    dirichlet_q5[psuf] = analytical.get_bc(grid_obj.faces_centers[face], permeability[0][0].reshape(3, 3),
                                                                            "quarter_five_spot", "dirichlet")
    
    # Add properties to cell data
    cell_data = {
        "linear": linear,
        "quadratic": quadratic,
        "permeability": list(permeability)
    }
    point_data = {
        "neumann_flag": neumann_flag,
        "dirichlet_flag": dirichlet_flag,

        "neumann_linear": neumann_l,
        "dirichlet_linear": dirichlet_l,
        "neumann_quadratic": neumann_q,
        "dirichlet_quadratic": dirichlet_q,
        
    }
    if "box" in file_name:
        cell_data["quarter_five_spot"] = quarter_five_spot
        point_data["neumann_quarter_five_spot"]   = neumann_q5
        point_data["dirichlet_quarter_five_spot"] = dirichlet_q5

    # Save modified mesh file
    file_name_without_extension = os.path.splitext(file_name)[0]
    output_file = os.path.join(output_dir, file_name_without_extension + ".vtk")
    mesh_out = meshio.Mesh(mesh.points, mesh.cells, cell_data=cell_data, point_data=point_data)
    meshio.write(output_file, mesh_out)

# Define directories
input_dir = "./pure_mesh"
output_dir = "./altered_mesh"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create temp output to create temporary files
temp_output_dir = "./temp_output/"

# Iterate over files in input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".msh"):  # Adjust file extensions if needed
        file_path = os.path.join(input_dir, file_name)
        process_mesh_file(file_name, file_path, output_dir, temp_output_dir)

# Remove temp output directory
print("Processing completed.")
