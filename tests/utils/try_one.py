import ninpol
import meshio
import numpy as np
import sys
import os
import memory_profiler
# add this file directory to the path at __file__
sys.path.append(os.path.dirname(__file__))
mesh_dir = "./altered/"

def load_mesh(mesh_dir, filename):
    msh = meshio.read(os.path.join(mesh_dir, filename))

def process_mesh(interpolador, msh):
    args = interpolador.process_mesh(msh)

def build_grid(interpolador, args):
    grid_obj = ninpol.grid.Grid(*args[:9])
    grid_obj.build(*args[9:])

def load_process_build(interpolador, mesh_dir, filename):
    interpolador.load_mesh(os.path.join(mesh_dir, filename))

def interpolate(interpolador, points):
    interpolador.interpolate(points, "linear", "inv_dist", "ctp")

mesh_file = "./altered/box1.vtk"
msh = meshio.read(mesh_file)
interpolador = ninpol.Interpolator()
interpolador.load_mesh(mesh_file)
point_coords = np.asarray(interpolador.grid_obj.point_coords)
points = np.arange(point_coords.shape[0])
measure = np.asarray(interpolador.interpolate(points, 'linear', 'inv_dist', "ctp"))
