import numpy as np
import sys
import os

ref = None
point_list = None
kdtree = None
pressure = None

def linear(centroid):
    return centroid[:, 0] + centroid[:, 1] + centroid[:, 2]

def quadratic(centroid):
    return centroid[:, 0]**2 + centroid[:, 1]**2 + centroid[:, 2]**2

def buildKDTree(point_list):
    from scipy.spatial import cKDTree
    return cKDTree(point_list)

def quarter_five_spot(centroid, is_box=True):
    if not is_box:
        return 0.0
    
    global ref
    global point_list
    global kdtree
    global pressure
    
    if ref is None:
        reference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "qfive_spot_ref.npy")

        ref = np.load(reference_path, allow_pickle=True).item()
        point_list = np.array([ref['x'], ref['y'], ref['z']]).T
        pressure = ref['p']

        kdtree = buildKDTree(point_list)

    # Find the nearest point in the reference data
    distance, idx = kdtree.query(centroid, k = 64)
    
    
    distance = 1.0 / distance
    ret_pressure = np.zeros_like(centroid[:, 0])
    for i in range(64):
        ret_pressure += pressure[idx[:, i]] * distance[:, i]
    ret_pressure /= np.sum(distance, axis=1)
    # Where pressure is NaN at index i, set it = pressure[idx[i, 0]]
    ret_pressure[np.isnan(ret_pressure)] = pressure[idx[np.isnan(ret_pressure), 0]]
    return ret_pressure


