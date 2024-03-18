import os
import meshio
import numpy as np
import analytical

# Move current path to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def calculate_properties(centroids):
    # Calculate properties for each centroid using the analytical package
    linear     = analytical.linear(centroids)
    quadratic           = analytical.quadratic(centroids)
    quarter_five_spot   = analytical.quarter_five_spot(centroids)

    return linear, quadratic, quarter_five_spot


def process_mesh_file(file_name, file_path, output_dir, temp_output_dir):

    print("Processing file:", file_name)

    # Read mesh file using meshio
    mesh = meshio.read(file_path)
    # Write in a temporary output directory
    meshio.write(temp_output_dir + "foo.vtk", mesh)

    # Read mesh file using meshio
    mesh = meshio.read(temp_output_dir + "foo.vtk")
    
    mesh.points = mesh.points.astype(float)

    n_cells = 0
    connectivity = None
    linear = []
    quadratic = []
    quarter_five_spot = []
    for cell_type in mesh.cells:
        n_cells = len(cell_type.data)
        cell_type.data = cell_type.data.astype(int)
        connectivity = cell_type.data

        centroids = np.mean(mesh.points[connectivity], axis=1)
        lp, q, q5 = calculate_properties(centroids)
        linear.append(lp)
        quadratic.append(q)
        quarter_five_spot.append(q5)


    
    

    # Add properties to cell data
    cell_data = {
        "linear": linear,
        "quadratic": quadratic,
        "quarter_five_spot": quarter_five_spot
    }

    # Save modified mesh file
    file_name_without_extension = os.path.splitext(file_name)[0]
    output_file = os.path.join(output_dir, file_name_without_extension + ".vtk")
    mesh_out = meshio.Mesh(mesh.points, mesh.cells, cell_data=cell_data)
    meshio.write(output_file, mesh_out)

# Define directories
input_dir = "./pure"
output_dir = "./altered"

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
