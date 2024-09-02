import numpy as np
import sys
import os

ref = None
point_list = None
kdtree = None
pressure = None

def calculate_neumann(expression, centroid, normal, permeability):
    import sympy as spy
    # Parse expression with sympy
    x, y, z = spy.symbols('x y z')
    expr = spy.sympify(expression.replace('^', '**'))

    # Get the gradient of the expression
    grad_expr = spy.Matrix([spy.diff(expr, var) for var in (x, y, z)])

    # Calculate gradient at centroid
    centroid_dict = {
                     x: centroid[:, 0] if len(centroid.shape) > 1 else centroid[0], 
                     y: centroid[:, 1] if len(centroid.shape) > 1 else centroid[1],
                     z: centroid[:, 2] if len(centroid.shape) > 1 else centroid[2]
                     }
    grad_at_centroid = grad_expr.subs(centroid_dict)

    # Define the permeability matrix
    K = spy.Matrix(permeability)

    # Calculate K dot grad u
    K_grad_u = K * grad_at_centroid

    # Calculate (-K dot grad u) dot n
    normal_vector = spy.Matrix(normal)
    neumann_value = (-K_grad_u).dot(normal_vector)

    return float(neumann_value)

def calculate_source(expression, centroid, permeability):
    import sympy as spy
    # Parse expression with sympy
    x, y, z = spy.symbols('x y z')
    expr = spy.sympify(expression.replace('^', '**'))

    # Source = -div(K grad u)
    # Get the gradient of the expression
    grad_expr = spy.Matrix([spy.diff(expr, var) for var in (x, y, z)])

    # Calculate gradient at centroid
    centroid_dict = {
                        x: centroid[:, 0] if len(centroid.shape) > 1 else centroid[0], 
                        y: centroid[:, 1] if len(centroid.shape) > 1 else centroid[1],
                        z: centroid[:, 2] if len(centroid.shape) > 1 else centroid[2]
                    }
    
    grad_at_centroid = grad_expr.subs(centroid_dict)

    # Define the permeability matrix
    K = spy.Matrix(permeability)

    # Calculate K dot grad u
    K_grad_u = K * grad_at_centroid

    # Calculate -div(K grad u)
    div_K_grad_u = sum([-spy.diff(K_grad_u[i], var) for i, var in enumerate((x, y, z))])

    return float(div_K_grad_u)


def get_bc(centroid, permeability, function, type, **args):
    centroid = np.asarray(centroid)
    if function == "linear":
        if type == "neumann":
            return 0.
        elif type == "dirichlet":
            return linear(centroid)
    elif function == "quadratic":
        if type == "neumann":
            args["normal"] = np.asarray(args["normal"])
            return calculate_neumann("x^2 + y^2 + z^2", 
                                     centroid, args["normal"], 
                                     permeability)
        elif type == "dirichlet":
            return quadratic(centroid)
    elif function == "quarter_five_spot":
        if type == "neumann":
            return 0.
        elif type == "dirichlet":
            return quarter_five_spot(np.array([centroid]))
    elif function == "u":
        if type == "neumann":
            return calculate_neumann("1 + sin(pi * x) * sin(pi * (y + 1/2)) * sin(pi * (z + 1/3))", 
                                     centroid, args["normal"], 
                                     permeability)
        elif type == "dirichlet":
            return u(centroid)

def get_source(centroid, permeability, function):
    centroid = np.asarray(centroid)
    if function == "linear":
        return calculate_source("x + y + z", centroid, permeability)
    elif function == "quadratic":
        return calculate_source("x^2 + y^2 + z^2", centroid, permeability)
    elif function == "u":
        return calculate_source("1 + sin(pi * x) * sin(pi * (y + 1/2)) * sin(pi * (z + 1/3))", centroid, permeability)
    elif function == "quarter_five_spot":
        return np.zeros_like(centroid[:, 0] if len(centroid.shape) > 1 else centroid[0])

def linear(centroid):
    if len(centroid.shape) == 1:
        return centroid[0] + centroid[1] + centroid[2]
    return centroid[:, 0] + centroid[:, 1] + centroid[:, 2]

def u(centroid):
    if len(centroid.shape) == 1:
        return 1 + np.sin(centroid[0] * np.pi) * np.sin((centroid[1] + 1/2) * np.pi) * np.sin((centroid[2] + 1/2) * np.pi)
    return 1 + np.sin(centroid[:, 0] * np.pi) * np.sin((centroid[:, 1] + 1/2) * np.pi) * np.sin((centroid[:, 2] + 1/2) * np.pi)

    
def quadratic(centroid):
    if len(centroid.shape) == 1:
        return centroid[0]**2 + centroid[1]**2 + centroid[2]**2
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


