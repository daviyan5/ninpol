"""
This file contains the "Interpolator" class definition, for interpolating mesh unknows. 
"""
import numpy as np
import time
import meshio
import scipy.sparse as sp

from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from cython.parallel import prange


DTYPE_I = np.int64
DTYPE_F = np.float64


cdef class Interpolator:

    def __cinit__(self, str name = "interpolator", int logging = False):
        # Load point-ordering.yaml 
        import yaml
        import os

        cdef:
            str current_directory = os.path.dirname(os.path.abspath(__file__))
            str current_parent_directory = os.path.dirname(current_directory)
            str point_ordering_path = os.path.join(current_parent_directory, "utils", "point_ordering.yaml")
        
        
        with open(point_ordering_path, 'r') as f:
            self.point_ordering = yaml.load(f, Loader=yaml.FullLoader)

        self.is_grid_initialized = False

        self.supported_methods = {
            "idw": inverse_distance,
            "gls": GLS
        }

        self.variable_to_index = {
            "points": {},
            "cells": {}
        }

        self.types_per_dimension = {
            0: ["vertex"],
            1: ["line"],
            2: ["triangle", "quad"],
            3: ["tetra", "hexahedron", "wedge", "pyramid"]
        }
        self.cells_data = np.zeros((1, 1), dtype=DTYPE_F)
        self.cells_data_dimensions = np.zeros(1, dtype=DTYPE_I)
        self.points_data = np.zeros((1, 1), dtype=DTYPE_F)
        self.points_data_dimensions = np.zeros(1, dtype=DTYPE_I)

        self.logging = logging
        self.logger  = Logger(name)

    
    def load_mesh(self, str filename = ""):
        if filename == "":
            raise ValueError("filename for the mesh must be provided.")
        # Loads a mesh from a file
        
        if self.logging:
            self.logger.log(f"Loading mesh from {filename}", "INFO")

        self.mesh_obj = meshio.read(filename)

        cdef tuple args = self.process_mesh(self.mesh_obj)

        self.grid_obj = Grid(*args)

        self.grid_obj.build()
        self.grid_obj.load_point_coords(self.mesh_obj.points.astype(DTYPE_F))
        self.grid_obj.calculate_centroids()
        self.grid_obj.calculate_normal_faces()
       
        if self.mesh_obj.cell_data:
            self.load_cell_data()
        else:
            self.cells_data = np.zeros((1, 1), dtype=DTYPE_F)
            self.cells_data_dimensions = np.zeros(1, dtype=DTYPE_I)

        if self.mesh_obj.point_data:
            self.load_point_data()
        else:
            self.points_data = np.zeros((1, 1), dtype=DTYPE_F)
            self.points_data_dimensions = np.zeros(1, dtype=DTYPE_I)

        self.is_grid_initialized = True

        if self.logging:
            self.logger.log(f"Mesh loaded successfully: {self.grid_obj.n_points} points", "INFO")
            if self.grid_obj.n_points < 10000:
                self.logger.json("grid", self.grid_obj.get_data())

                self.logger.json("interpolator", self.get_dict())
                self.logger.log("Grid loaded successfully", "INFO")
            else:
                self.logger.log("Grid too large to be logged", "WARN")
        
    def process_mesh(self, object mesh):
        cdef:
            int dim = mesh.points.shape[1]
            int n_points = mesh.points.shape[0]
            int n_elems = 0

            int nfael_e
            list lnofa_e, lpofa_e
            list face
            int n_point, point

            int npoed_e
            list lpoed_e
            list edge

            str elem_type_str
            dict elem_type
            int type_index, i, j

            DTYPE_I_t[::1] npoel       = np.ones(NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, dtype=DTYPE_I) * -1

            DTYPE_I_t[::1] nfael       = np.ones( NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, dtype=DTYPE_I) * -1

            DTYPE_I_t[:, ::1] lnofa    = np.ones((NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, 
                                                  NinpolSizes.NINPOL_MAX_FACES_PER_ELEMENT), dtype=DTYPE_I) * -1

            DTYPE_I_t[:, :, ::1] lpofa = np.ones((NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, 
                                                  NinpolSizes.NINPOL_MAX_FACES_PER_ELEMENT, 
                                                  NinpolSizes.NINPOL_MAX_POINTS_PER_FACE), dtype=DTYPE_I) * -1

            DTYPE_I_t[::1] nedel       = np.ones( NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, dtype=DTYPE_I) * -1

            DTYPE_I_t[:, :, ::1] lpoed = np.ones((NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, 
                                                  NinpolSizes.NINPOL_MAX_EDGES_PER_ELEMENT, 2), dtype=DTYPE_I) * -1


        
        faces_key = "faces"
        if dim == 2:
            faces_key = "edges"
        
        for elem_type_str, elem_type in self.point_ordering["elements"].items():
            type_index = elem_type["element_type"]
            npoel[type_index] = elem_type["number_of_points"]
            
            if elem_type_str not in self.types_per_dimension[dim]:
                continue

            nfael_e = len(elem_type[faces_key] if faces_key in elem_type else [])
            lnofa_e = [len(face) for face in elem_type[faces_key]] if faces_key in elem_type else []
            lpofa_e = elem_type[faces_key] if faces_key in elem_type else []

            npoed_e = len(elem_type["edges"] if "edges" in elem_type else [])
            lpoed_e = elem_type["edges"] if "edges" in elem_type else []

            
            
            nfael[type_index] = nfael_e
            if "faces" in elem_type:
                for i, n_point in enumerate(lnofa_e):
                    lnofa[type_index, i] = n_point
                    
                for i, face in enumerate(lpofa_e):
                    for j, point in enumerate(face):
                        lpofa[type_index, i, j] = point

            nedel[type_index] = npoed_e
            
            if "edges" in elem_type:
                for i, edge in enumerate(lpoed_e):
                    for j, point in enumerate(edge):
                        lpoed[type_index, i, j] = point


        for CellBlock in mesh.cells:
            elem_type_str = CellBlock.type
            if elem_type_str not in self.types_per_dimension[dim]:
                continue
            n_elems += len(CellBlock.data)

        cdef: 
            dict point_tag_to_index = {}
            dict cell_tag_to_index = {}

            int point_index = 0
            int cell_index = 0    

            int elem_type_index
            
            DTYPE_I_t[:, ::1] connectivity   = np.ones((n_elems, NinpolSizes.NINPOL_MAX_POINTS_PER_ELEMENT), dtype=DTYPE_I) * -1
            DTYPE_I_t[::1] element_types     = np.ones(n_elems, dtype=DTYPE_I) * -1
        
        for CellBlock in mesh.cells:
            elem_type_str = CellBlock.type

            if elem_type_str not in self.types_per_dimension[dim]:
                continue
            
            elem_type_index = self.point_ordering["elements"][elem_type_str]["element_type"]

            for i, cell in enumerate(CellBlock.data):
                # Sort cell points according to the following point ordering:
                # 1 - x-axis increasing
                for j, point in enumerate(cell):
                    connectivity[cell_index, j] = point
                element_types[cell_index] = elem_type_index
                cell_index += 1

        return (dim, 
                n_elems, n_points, 
                npoel,
                nfael, lnofa, lpofa, 
                nedel, lpoed,
                connectivity, element_types)

    
    def load_data(self, data_dict, data_type):
    # Data must be homogeneous
    # Only supports scalar and vector data. Each vector must have the same dimension

        cdef:
            int n_variables = len(data_dict)
            int n_elements = self.grid_obj.n_elems if data_type == "cells" else self.grid_obj.n_points
            int max_shape = 1, cur_shape

            int i, j

            int elem_or_point
            str variable

            int index = 0

        dimensions_array = np.zeros(n_variables, dtype=DTYPE_I)
        for variable in data_dict:  # Iterating over variables (e.g "pressure")
            cur_shape = data_dict[variable].shape[1] if len(data_dict[variable].shape) > 1 else 1
            max_shape = max(max_shape, cur_shape)
        
            self.variable_to_index[data_type][variable] = index
            dimensions_array[index] = cur_shape
            index += 1

        data_array = np.zeros((n_variables, n_elements * max_shape), dtype=DTYPE_F)

        cdef:
            int aux_index

        
        for variable in data_dict:                              # Iterating over variables (e.g "pressure")
            index = self.variable_to_index[data_type][variable]
            cur_shape = dimensions_array[index]
                
            
            for elem_or_point in range(n_elements):             # Iterating over elements or points
                if cur_shape == 1:
                    if len(data_dict[variable].shape) == 1:
                        data_array[index, elem_or_point] = data_dict[variable][elem_or_point]
                    else:
                        data_array[index, elem_or_point] = data_dict[variable][elem_or_point][0]
                else:
                    for j in range(cur_shape):
                        aux_index = elem_or_point * cur_shape + j
                        data_array[index, aux_index] = data_dict[variable][elem_or_point][j]

        if data_type == "cells":
            self.cells_data_dimensions = dimensions_array
            self.cells_data = data_array
        else:
            self.points_data_dimensions = dimensions_array
            self.points_data = data_array

    def load_cell_data(self):
        # cell_data_dict will save each variable separately for element type. We dont want that.
        
        cdef:
            int dim = self.grid_obj.dim
            dict cell_data_dict = self.mesh_obj.cell_data_dict
            dict cell_data = {}
            str variable
            str element_type
            list aux_array  
            int elem

        for variable in cell_data_dict:
            aux_array = []
            for element_type in cell_data_dict[variable]:
                if element_type not in self.types_per_dimension[dim]:
                    continue
                aux_array.extend(cell_data_dict[variable][element_type])    # Works because the cells are (should be) indexed in order

            cell_data[variable] = np.array(aux_array)
            if variable == "permeability":
                cell_data["diff_mag"] = self.compute_diffusion_magnitude(cell_data["permeability"])

        self.load_data(cell_data, "cells")

    def load_point_data(self):
        self.load_data(self.mesh_obj.point_data, "points")

    cdef DTYPE_F_t[::1] compute_diffusion_magnitude(self, DTYPE_F_t[:, ::1] permeability):
        cdef:
            int nvols = len(permeability)
            DTYPE_F_t[:, :, ::1] Ks = np.reshape(permeability, (nvols, 3, 3))                                    
            cnp.ndarray detKs = np.linalg.det(Ks)                                                          
            cnp.ndarray trKs  = np.trace(Ks, axis1=1, axis2=2)                                                   
            DTYPE_F_t[::1] diff_mag = (1 - (3 * (detKs ** (1 / 3)) / trKs)) ** 2      

        return diff_mag   

    def get_dict(self):
        cdef dict data_dict

        data_dict = {
            "point_ordering": self.point_ordering,

            "variable_to_index": self.variable_to_index,

            "cells_data": self.cells_data,
            "cells_data_dimensions": self.cells_data_dimensions,

            "points_data": self.points_data,
            "points_data_dimensions": self.points_data_dimensions,
        }

        # Convert memoryviews to np arrays
        data_dict["cells_data"] = np.asarray(data_dict["cells_data"])
        data_dict["cells_data_dimensions"] = np.asarray(data_dict["cells_data_dimensions"])

        data_dict["points_data"] = np.asarray(data_dict["points_data"])
        data_dict["points_data_dimensions"] = np.asarray(data_dict["points_data_dimensions"])
        # Print type of every variable
        
        return data_dict

    def get_data(self, str data_type, DTYPE_I_t[::1] index, str variable):
        cdef int data_index = 0
        if data_type == "cells":
            if variable not in self.variable_to_index["cells"]:
                raise ValueError(f"Variable '{variable}' not found in cells data.")
            data_index = self.variable_to_index["cells"][variable]
            return np.asarray(self.cells_data[data_index])[index]
        else:
            if variable not in self.variable_to_index["points"]:
                raise ValueError(f"Variable '{variable}' not found in points data.")
            data_index = self.variable_to_index["points"][variable]
            return np.asarray(self.points_data[data_index])[index]

    def interpolate(self, str variable,  str method, int return_value = False, DTYPE_I_t[::1] target_points = np.array([], dtype=DTYPE_I)):
        
        if not self.is_grid_initialized:
            raise ValueError("Grid not initialized. Please load a mesh first.")
        
        if method not in self.supported_methods:
            raise ValueError(f"Method '{method}' not supported. Supported methods are: {list(self.supported_methods.keys())}")

        if len(target_points) == 0:
            target_points = np.arange(self.grid_obj.n_points, dtype=DTYPE_I)

        cdef:
            int data_index = 0
            int data_dimension = 1
            DTYPE_F_t[::1] source_data = np.zeros(1, dtype=DTYPE_F)
            str source_type = "cells"

       
        if variable not in self.variable_to_index["cells"]:
            raise ValueError(f"Variable '{variable}' not found in cells data.")

        data_index     = self.variable_to_index["cells"][variable]
        source_data    = self.cells_data[data_index]
        data_dimension = self.cells_data_dimensions[data_index]
        
        cdef: 
            DTYPE_F_t[:, ::1] weights 
            DTYPE_I_t[:, ::1] connectivity_idx
            DTYPE_F_t[::1] neumann_ws

        if self.logging:
            self.logger.log(f"Interpolating variable '{variable}' using method '{method}'", "INFO")

        # The value of the interpolation is the product of the lines of the 
        # weight matrix with the lines of the connectivity matrix, and the sum of the results for each line
        weights, connectivity_idx, neumann_ws = self.prepare_interpolator(method, variable, data_dimension, target_points)
        
        cdef:
            int i, j, k
            DTYPE_F_t[:, ::1] result = np.zeros((len(target_points), data_dimension), dtype=DTYPE_F)
            int n_target = len(target_points)
            int n_elems = self.grid_obj.n_elems * data_dimension
            int n_columns = self.grid_obj.MX_ELEMENTS_PER_POINT * data_dimension
            int index, elem

            int size = 0

            DTYPE_I_t[::1] rows = np.ones(n_target * n_columns, dtype=DTYPE_I) * -1
            DTYPE_I_t[::1] cols = np.ones(n_target * n_columns, dtype=DTYPE_I) * -1
            DTYPE_F_t[::1] data = np.ones(n_target * n_columns, dtype=DTYPE_F) * -1    
            int use_threads = min(8, np.ceil(n_target / 800))

            int MX_ELEMENTS_PER_POINT = self.grid_obj.MX_ELEMENTS_PER_POINT
        
        cdef:
            dict data_dict
            
        if self.logging:
            self.logger.log(f"Interpolation prepared successfully", "INFO")
            data_dict = {
                "weights": np.asarray(weights),
                "connectivity_idx": np.asarray(connectivity_idx),
                "neumann_ws": np.asarray(neumann_ws)
            }
            self.logger.json(method + "_" + variable, data_dict)

        if return_value:
            
            # Multiply the weights by the connectivity, element-wise, and sum  
            for i in prange(n_target, nogil=True, schedule='static', num_threads=use_threads):
                for k in range(data_dimension):
                    for j in range(self.grid_obj.MX_ELEMENTS_PER_POINT):
                        index = i * n_columns + j * data_dimension + k
                        elem = connectivity_idx[i, j]
                        if elem == -1:
                            continue
                        result[i, k] += weights[i, j] * source_data[elem * data_dimension + k]

            # Remove unnecessary dimensions
            return np.asarray(np.squeeze(result)), np.asarray(neumann_ws)

        else:
            # Convert weights from (n_target, n_columns) to (n_target, n_elems) using sparse matrix
            size = 0
            index = 0
            for i in range(n_target): 
                for j in range(MX_ELEMENTS_PER_POINT):
                    for k in range(data_dimension):
                        if connectivity_idx[i, j] == -1:
                            continue
                        rows[index] = i
                        cols[index] = connectivity_idx[i, j] * data_dimension + k
                        data[index] = weights[i, j * data_dimension + k] + neumann_ws[i]
                        index += 1

            # Get smallest index where data is -1
            size = np.where(np.asarray(data) == -1)[0][0]

            weights_sparse = sp.csr_matrix((np.asarray(data)[:size], 
                                           (np.asarray(rows)[:size], np.asarray(cols)[:size])), 
                                           shape=(n_target, n_elems))
            weights_sparse.eliminate_zeros()
            
            return weights_sparse, np.asarray(neumann_ws)

    cdef tuple prepare_interpolator(self, str method, str variable,
                                    const int data_dimension, 
                                    const DTYPE_I_t[::1] target_points):
        
        
        cdef: 
            int n_target = len(target_points)
            int n_columns = self.grid_obj.MX_ELEMENTS_PER_POINT * data_dimension
            int dim = self.grid_obj.dim

            int nm_flag_index, nm_index
            int permeability_index, diff_mag_index

            DTYPE_I_t[:, ::1] connectivity_idx  = np.ones((n_target, self.grid_obj.MX_ELEMENTS_PER_POINT), dtype=DTYPE_I) * -1
            DTYPE_F_t[:, ::1] weights           = np.zeros((n_target, n_columns), dtype=DTYPE_F)

            DTYPE_F_t[:, ::1] target_coordinates = np.asarray(self.grid_obj.point_coords)[target_points]
            DTYPE_F_t[:, ::1] source_coordinates = self.grid_obj.centroids 

            int point, elem, first, last, i, j, k
            int use_threads = min(8, np.ceil(n_target / 800))

            DTYPE_I_t[::1] in_points, nm_points
            DTYPE_F_t[:, :, ::1] permeability = np.zeros((self.grid_obj.n_elems, dim, dim), dtype=DTYPE_F)
            DTYPE_F_t[::1] diff_mag   = np.zeros(self.grid_obj.n_elems, dtype=DTYPE_F)
            DTYPE_F_t[::1] neumann    = np.zeros(self.grid_obj.n_points, dtype=DTYPE_F)
            DTYPE_F_t[::1] neumann_ws = np.zeros(n_target, dtype=DTYPE_F)

        
        # Populate connectivity_idx
        omp_set_num_threads(use_threads)
        for i in prange(n_target, nogil=True, schedule='static', num_threads=use_threads):
            point = target_points[i]
            first = self.grid_obj.esup_ptr[point]
            last = self.grid_obj.esup_ptr[point + 1]
            for j, elem in enumerate(self.grid_obj.esup[first:last]):
                connectivity_idx[i, j] = elem

        if method == "gls":
            
            in_points     = np.where(np.asarray(self.grid_obj.boundary_points) == 0)[0]
            
            nm_flag_index = self.variable_to_index["points"]["neumann_flag"]
            nm_index      = self.variable_to_index["points"]["neumann" + "_" + variable]

            permeability_index = self.variable_to_index["cells"]["permeability"]
            diff_mag_index     = self.variable_to_index["cells"]["diff_mag"]

            nm_points     = np.where(np.asarray(self.points_data[nm_flag_index]) == 1)[0]
            
            permeability  = np.reshape(self.cells_data[permeability_index], 
                                      (self.grid_obj.n_elems, dim, dim))

            diff_mag      = self.cells_data[diff_mag_index]
            neumann       = self.points_data[nm_index]

            
            self.supported_methods[method](
                self.grid_obj, 
                in_points, nm_points,
                permeability, diff_mag,
                neumann,
                weights, neumann_ws
            )

        if method == "idw":
            self.supported_methods[method](
                dim,
                target_coordinates, source_coordinates, 
                connectivity_idx, weights
            )
        
        return weights, connectivity_idx, neumann_ws


