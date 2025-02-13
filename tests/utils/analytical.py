import numpy as np
import sympy as sp
import pickle
import meshio
import ninpol

def evaluate_expr(expr_str):
    # Define the variables
    x, y, z = sp.symbols('x y z')
    v = (x, y, z)

    u_expr = sp.sympify(expr_str)

    K11, K12, K13 = sp.symbols('K11 K12 K13')
    K21, K22, K23 = sp.symbols('K21 K22 K23')
    K31, K32, K33 = sp.symbols('K31 K32 K33')

    K = sp.Matrix([[K11, K12, K13], 
                   [K21, K22, K23], 
                   [K31, K32, K33]])

    grad_u = sp.Matrix([sp.diff(u_expr, var) for var in v])

    K_grad_u = K * grad_u
    n1, n2, n3 = sp.symbols('n1 n2 n3')
    normal = sp.Matrix([n1, n2, n3])

    neu_expr = -K_grad_u.dot(normal)

    return neu_expr

def lambdify_expr(neu_expr):
    # Define the symbols to be substituted
    x, y, z = sp.symbols('x y z')
    n1, n2, n3 = sp.symbols('n1 n2 n3')
    K11, K12, K13 = sp.symbols('K11 K12 K13')
    K21, K22, K23 = sp.symbols('K21 K22 K23')
    K31, K32, K33 = sp.symbols('K31 K32 K33')

    # Lambdify the expression for fast numerical evaluation
    lambdified_func = sp.lambdify(
        (K11, K12, K13, K21, K22, K23, K31, K32, K33, n1, n2, n3, x, y, z),
        neu_expr,
        'numpy'
    )

    return lambdified_func

def evaluate_neumann(lambdified_func, K_vals, n_vals, x_vals, y_vals, z_vals):

    # Vectorize the input values to make use of numpy operations
    K_vals = np.array(K_vals)
    n_vals = np.array(n_vals)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    K11_vals = K_vals[:, 0, 0]
    K12_vals = K_vals[:, 0, 1]
    K13_vals = K_vals[:, 0, 2]
    K21_vals = K_vals[:, 1, 0]
    K22_vals = K_vals[:, 1, 1]
    K23_vals = K_vals[:, 1, 2]
    K31_vals = K_vals[:, 2, 0]
    K32_vals = K_vals[:, 2, 1]
    K33_vals = K_vals[:, 2, 2]

    n1_vals = n_vals[:, 0]
    n2_vals = n_vals[:, 1]
    n3_vals = n_vals[:, 2]

    # Use the lambdified function in a vectorized way for all data points
    results = lambdified_func(
        K11_vals, K12_vals, K13_vals, 
        K21_vals, K22_vals, K23_vals, 
        K31_vals, K32_vals, K33_vals, 
        n1_vals, n2_vals, n3_vals,
        x_vals, y_vals, z_vals
    )

    return np.array(results)



def clean_mesh(mesh):
    # Remove Non-Dimension cells
    dim = 1
    types_per_dimension = {
        0: ["vertex"],
        1: ["line"],
        2: ["triangle", "quad"],
        3: ["tetra", "hexahedron", "wedge", "pyramid"]
    }
    for key in mesh.cells:
        for dim_key in types_per_dimension:
            if key.type in types_per_dimension[dim_key]:
                dim = max(dim, dim_key)
    valid_cells = []
    for index, key in enumerate(mesh.cells_dict):
        if key in types_per_dimension[dim]:
            valid_cells.append(index)
    mesh.cells = [mesh.cells[i] for i in valid_cells]
    mesh.cell_data = {}
    return mesh

def l2norm_relative(measure, reference):
    sqr_sum = np.sum(reference ** 2)
    if sqr_sum == 0:
        return np.nan
    return np.sqrt(np.sum((measure - reference) ** 2) / sqr_sum)

def l2norm_array(measure, reference):
    sqr_sum = np.sum(reference ** 2)
    if sqr_sum == 0:
        return np.array([np.nan for _ in range(len(measure))])
    return np.sqrt(((measure - reference) ** 2) / sqr_sum) 


class BaseCase():
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression
        self.neu_expr   = evaluate_expr(expression)

    def assign_mesh_properties(self, mesh_filename):

        mesh = meshio.read(mesh_filename)
        mesh = clean_mesh(mesh)
        
        I = ninpol.Interpolator()
        
        I.load_mesh(mesh_obj=mesh)

        grid = I.grid
        permeability = [[] for _ in range(len(mesh.cells))]
        solution     = [[] for _ in range(len(mesh.cells))]

        for index in range(len(mesh.cells)):
            centroids = np.mean(mesh.points[mesh.cells[index].data], axis=1)
            K = self.calculate_K(len(mesh.cells[index].data), centroids)
            permeability[index] = K.reshape(-1, 9)

            solution[index] = self.solution(
                centroids[:, 0], centroids[:, 1], centroids[:, 2]
            )
        
        self.vols_solution = np.concatenate(solution)
        
        faces_normals = np.asarray(grid.normal_faces)

        boundary         = np.where(np.asarray(grid.boundary_faces))[0]
        self.boundary_points  = np.unique(np.asarray(grid.inpofa)[boundary].flatten())
        self.boundary_points  = self.boundary_points[self.boundary_points != -1]
        self.internal_points  = np.setdiff1d(np.arange(grid.n_points), self.boundary_points)

        rate = 1/2
        random_indexes  = np.random.choice(len(boundary), int(len(boundary) * rate), replace=False)
        dirichlet_faces = boundary[random_indexes]
        neumann_faces   = np.setdiff1d(boundary, dirichlet_faces)

        # A point can only be Dirichlet or Neumann. 
        # The point will belong to the Dirichlet set if the majority of the boundary faces that it is part of are Dirichlet
        # Otherwise, it will belong to the Neumann set
        
        points_values = np.zeros(grid.n_points)
        points_values[self.internal_points] = np.nan
        dirichlet_points = np.asarray(grid.inpofa)[dirichlet_faces].flatten()
        dirichlet_points = dirichlet_points[dirichlet_points != -1]
        points_values[dirichlet_points] += 1

        neumann_points = np.asarray(grid.inpofa)[neumann_faces].flatten()
        neumann_points = neumann_points[neumann_points != -1]
        points_values[neumann_points] -= 1

        dirichlet_points = np.where(points_values >= 0)[0]
        neumann_points = np.where(points_values < 0)[0]        

        dirichlet_flag = np.zeros(grid.n_points)
        dirichlet_flag[dirichlet_points] = 1

        neumann_flag = np.zeros(grid.n_points)
        neumann_flag[neumann_points] = 1

        P = np.asarray(grid.point_coords)
        dP = np.asarray(grid.point_coords)[dirichlet_points]
        
        dirichlet = np.zeros(grid.n_points)
        dirichlet[dirichlet_points] = self.solution(
            dP[:, 0], dP[:, 1], dP[:, 2]
        )
        
        neumann = np.zeros(grid.n_points)
        neumann_val_faces = np.zeros(grid.n_faces)
        lamdified_func = lambdify_expr(self.neu_expr)

        volumes_around_faces = np.asarray(grid.esuf)
        volumes_around_faces = volumes_around_faces[np.asarray(grid.esuf_ptr)[boundary]]
        
        permeability_volumes = self.calculate_K(grid.n_elems, np.asarray(grid.centroids))
        K_neumann = np.array(permeability_volumes)[volumes_around_faces]
        K_neumann = K_neumann.reshape(-1, 3, 3)
        n_neumann = faces_normals[boundary]
        faces_centers = np.asarray(grid.faces_centers)
        x_neumann = faces_centers[boundary, 0]
        y_neumann = faces_centers[boundary, 1]
        z_neumann = faces_centers[boundary, 2]
        
        neumann_val_faces[boundary] = evaluate_neumann(
            lamdified_func, K_neumann, n_neumann, x_neumann, y_neumann, z_neumann
        )

        neumann[neumann_points] = np.array([np.mean(neumann_val_faces[grid.fsup[grid.fsup_ptr[point]:grid.fsup_ptr[point + 1]]]) for point in neumann_points])


        self.point_solution = self.solution(
            P[:, 0], P[:, 1], P[:, 2]
        )

        point_data = {
            "dirichlet_"      + self.name: dirichlet,
            "dirichlet_flag_" + self.name: dirichlet_flag,

            "neumann_"        + self.name: neumann,
            "neumann_flag_"   + self.name: neumann_flag
        }
        self.dirichlet_points = dirichlet_points
        cell_data = {
            "permeability": permeability,
            self.name: solution
        }
        self.mesh = meshio.Mesh(mesh.points, mesh.cells, point_data, cell_data)

    def evaluate(self, weights):
        # Return l2 norm of the error and l2 array of the error

        values      = weights.dot(self.vols_solution)
        values[self.dirichlet_points] = self.point_solution[self.dirichlet_points]

        internal_nodes = np.setdiff1d(np.arange(len(values)), self.boundary_points)
        error       = l2norm_relative(values[internal_nodes], self.point_solution[internal_nodes])
        error_array = l2norm_array(values, self.point_solution)
        
        return error, error_array
    
    

            

class LINCase(BaseCase):
    def __init__(self):
        super().__init__("LIN", "x + y + z")

    def calculate_K(self, n, centroids=None):
        Ku = np.array( [[1.0, 0.5, 0.0], 
                        [0.5, 1.0, 0.5], 
                        [0.0, 0.5, 1.0]])
        
        K = np.zeros((n, 3, 3))
        K[:, :, :] = Ku
        return K
    
    def solution(self, x, y, z):
        return x + y + z

class QUADCase(BaseCase):
    def __init__(self):
        super().__init__("QUAD", "x**2 + y**2 + z**2")

    def calculate_K(self, n, centroids=None):
        Ku = np.array( [[1.0, 0.5, 0.0], 
                        [0.5, 1.0, 0.5], 
                        [0.0, 0.5, 1.0]])
        
        K = np.zeros((n, 3, 3))
        K[:, :, :] = Ku
        return K
    
    def solution(self, x, y, z):
        return x**2 + y**2 + z**2

class FANcase(BaseCase):
    def __init__(self):
        super().__init__("FAN", "sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z)")

    def calculate_K(self, n, centroids=None):
        Ku = np.array(
            [[2464.36,    0.0, 1148.68], 
             [    0.0, 536.64,     0.0], 
             [1148.68,    0.0,  536.64]]
        )
        K = np.zeros((n, 3, 3))
        K[:, :, :] = Ku
        return K
    
    def solution(self, x, y, z):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)


class ALHcase(BaseCase):
    def __init__(self):
        super().__init__("ALH", "x**3 * y**2 * z + x * sin(2 * pi * x * z) * sin(2 * pi * x * y) * sin(2 * pi * z)")

    def calculate_K(self, n, centroids=None):
        # Initialize the K array with shape (n, 3, 3) where n is the length of x, y, z
        K = np.zeros((n, 3, 3))

        x = centroids[:, 0]
        y = centroids[:, 1]
        z = centroids[:, 2]

        K[:, 0, 0] = y ** 2 + z ** 2 + 1
        K[:, 0, 1] = -x * y
        K[:, 0, 2] = -x * z
        
        K[:, 1, 0] = -y * x
        K[:, 1, 1] = x ** 2 + z ** 2 + 1
        K[:, 1, 2] = -y * z
        
        K[:, 2, 0] = -z * x
        K[:, 2, 1] = -z * y
        K[:, 2, 2] = x ** 2 + y ** 2 + 1

        return K

    def solution(self, x, y, z):
        return (x ** 3) * (y ** 2) * z + x * np.sin(2 * np.pi * x * z) * np.sin(2 * np.pi * x * y) * np.sin(2 * np.pi * z)