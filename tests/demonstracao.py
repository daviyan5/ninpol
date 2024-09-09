import numpy as np

import ninpol
import meshio

def linear(centroid):
    return centroid[:, 0] + centroid[:, 1] + centroid[:, 2]

mesh_file = "./tests/utils/altered_mesh/benchtetra0.vtk"
interpolador = ninpol.Interpolator(mesh_file)

msh = meshio.read(mesh_file)

# Interpolando todos os pontos
weights, _ = interpolador.interpolate("linear", "gls")

print(weights, _)