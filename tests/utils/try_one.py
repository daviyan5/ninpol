import ninpol
import meshio
import numpy as np
import sys
import os
import memory_profiler
import analytical


mesh_dir = "./altered_mesh/"

def load_mesh(mesh_dir, filename):
    msh = meshio.read(os.path.join(mesh_dir, filename))
    return msh

def process_mesh(interpolador, msh):
    args = interpolador.process_mesh(msh)
    return args

def build_grid(interpolador, args):
    grid_obj = ninpol.Grid(*args[:9])
    grid_obj.build(*args[9:])
    return grid_obj

def load_process_build(interpolador, mesh_dir, filename):
    interpolador.load_mesh(os.path.join(mesh_dir, filename))

def interpolate(interpolador, points):
    interpolador.interpolate(points, "quarter_five_spot", "inv_dist", "ctp")

def l2norm_relative(measure, reference):
    sqr_sum = np.sum(reference ** 2)
    return np.sqrt(np.sum((measure - reference) ** 2) / sqr_sum)

mesh_file = os.path.join(os.path.dirname(__file__), "altered_mesh", "box4.vtk")
msh = meshio.read(mesh_file)
interpolador = ninpol.Interpolator()
interpolador.load_mesh(mesh_file)
point_coords = np.asarray(interpolador.grid_obj.point_coords)
points = np.arange(point_coords.shape[0])
measure = np.asarray(interpolador.interpolate(points, 'quarter_five_spot', 'inv_dist', "ctp"))
