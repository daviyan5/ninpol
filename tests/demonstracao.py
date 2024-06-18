import numpy as np

import ninpol
import meshio

def linear(centroid):
    return centroid[:, 0] + centroid[:, 1] + centroid[:, 2]

mesh_file = "./tests/utils/altered_mesh/box0.vtk"
interpolador = ninpol.Interpolator(mesh_file)

msh = meshio.read(mesh_file)
print(msh)

# Interpolando todos os pontos
weights = interpolador.interpolate("linear", "gls")

vals = interpolador.interpolate("linear", "gls", return_value=True)
