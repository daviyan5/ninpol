"""
This file contains the "Interpolator" class definition, for interpolating mesh unknows. 
"""
import numpy as np
import meshio
import scipy.sparse as sp

from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from cython.parallel import prange

DTYPE_I = np.int64
DTYPE_F = np.float64


cdef class Interpolator:

    def __cinit__(self):
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
            "inv_dist": distance_inverse
        }

        self.variable_to_index = {
            "points": {},
            "cells": {}
        }
        self.cells_data = np.zeros((1, 1), dtype=DTYPE_F)
        self.cells_data_dimensions = np.zeros(1, dtype=DTYPE_I)
        self.points_data = np.zeros((1, 1), dtype=DTYPE_F)
        self.points_data_dimensions = np.zeros(1, dtype=DTYPE_I)
    
    def load_mesh(self, str filename = ""):
        if filename == "":
            raise ValueError("Either 'filename' or 'mesh' must be provided.")
        # Loads a mesh from a file
        
        self.mesh_obj = meshio.read(filename)

        cdef tuple args = self.process_mesh(self.mesh_obj)

        self.grid_obj = Grid(*args)

        self.grid_obj.build()
        
        self.grid_obj.load_point_coords(self.mesh_obj.points.astype(DTYPE_F))
        self.grid_obj.calculate_cells_centroids()
       
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

            DTYPE_I_t[::1] npoel       = np.ones(NINPOL_NUM_ELEMENT_TYPES, dtype=DTYPE_I) * -1

            DTYPE_I_t[::1] nfael       = np.ones( NINPOL_NUM_ELEMENT_TYPES, dtype=DTYPE_I) * -1

            DTYPE_I_t[:, ::1] lnofa    = np.ones((NINPOL_NUM_ELEMENT_TYPES, 
                                                  NINPOL_MAX_FACES_PER_ELEMENT), dtype=DTYPE_I) * -1

            DTYPE_I_t[:, :, ::1] lpofa = np.ones((NINPOL_NUM_ELEMENT_TYPES, 
                                                  NINPOL_MAX_FACES_PER_ELEMENT, 
                                                  NINPOL_MAX_POINTS_PER_FACE), dtype=DTYPE_I) * -1

            DTYPE_I_t[::1] nedel       = np.ones( NINPOL_NUM_ELEMENT_TYPES, dtype=DTYPE_I) * -1

            DTYPE_I_t[:, :, ::1] lpoed = np.ones((NINPOL_NUM_ELEMENT_TYPES, 
                                                  NINPOL_MAX_EDGES_PER_ELEMENT, 2), dtype=DTYPE_I) * -1


        
        faces_key = "faces"
        if dim == 2:
            faces_key = "edges"
        
        for elem_type in self.point_ordering["elements"].values():
            type_index = elem_type["element_type"]
            npoel[type_index] = elem_type["number_of_points"]

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
            n_elems += len(CellBlock.data)

        cdef: 
            dict point_tag_to_index = {}
            dict cell_tag_to_index = {}

            int point_index = 0
            int cell_index = 0    

            int elem_type_index
            
            DTYPE_I_t[:, ::1] connectivity   = np.ones((n_elems, NINPOL_MAX_POINTS_PER_ELEMENT), dtype=DTYPE_I) * -1
            DTYPE_I_t[::1] element_types     = np.ones(n_elems, dtype=DTYPE_I) * -1
        
        for CellBlock in mesh.cells:
            elem_type_str = CellBlock.type
            elem_type_index = self.point_ordering["elements"][elem_type_str]["element_type"]

            for i, cell in enumerate(CellBlock.data):
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
            dict cell_data_dict = self.mesh_obj.cell_data
            str variable
            cnp.ndarray element_type
            list aux_array  
            int index, elem

        for variable in cell_data_dict:
            aux_array = []
            index = 0
            for element_type in cell_data_dict[variable]:
                aux_array.extend(element_type)

            cell_data_dict[variable] = np.array(aux_array)

        self.load_data(cell_data_dict, "cells")

    def load_point_data(self):
        self.load_data(self.mesh_obj.point_data, "points")

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

    def interpolate(self, DTYPE_I_t[::1] target_points, str variable,  str method, int return_value = False):
        
        if not self.is_grid_initialized:
            raise ValueError("Grid not initialized. Please load a mesh first.")
        
        if method not in self.supported_methods:
            raise ValueError(f"Method '{method}' not supported. Supported methods are: {list(self.supported_methods.keys())}")


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
        
        cdef DTYPE_F_t[:, ::1] weights, connectivity_val, connectivity_idx

        # The value of the interpolation is the product of the lines of the 
        # weight matrix with the lines of the connectivity matrix, and the sum of the results for each line
        weights, connectivity_idx, connectivity_val = self.prepare_interpolator(method, source_data, data_dimension, target_points)


        if return_value:
            # Multiply the weights by the connectivity, element-wise, and sum
            cdef:
                int i
                DTYPE_F_t[:, ::1] unsummed_result = np.multiply(weights, connectivity_val)
                DTYPE_F_t[:, ::1] result = np.zeros((len(target_points), data_dimension), dtype=DTYPE_F)
                
            for i in range(data_dimension):
                result[:, i] = np.sum(unsummed_result[:, i::data_dimension], axis=1)

            # Remove unnecessary dimensions
            return np.squeeze(result)

        else:
            # Convert weights from (n_target, n_columns) to (n_target, n_elems) using sparse matrix
            cdef:
                int n_target = len(target_points)
                int n_elems = self.grid_obj.n_elems * data_dimension
                int n_columns = self.grid_obj.MX_ELEMENTS_PER_POINT * data_dimension
                int i, j, k
                int index

                DTYPE_I_t[::1] rows = np.zeros(n_target * n_columns, dtype=DTYPE_I)
                DTYPE_I_t[::1] cols = np.zeros(n_target * n_columns, dtype=DTYPE_I)   
                DTYPE_F_t[::1] data = np.zeros(n_target * n_columns, dtype=DTYPE_F)         
                int use_threads = min(8, np.ceil(n_target / 800))

            omp_set_num_threads(use_threads)
            for i in prange(n_target, nogil=True, schedule='static', num_threads=use_threads):
                for j in range(self.grid_obj.MX_ELEMENTS_PER_POINT):
                    for k in range(data_dimension):
                        index = i * n_columns + j * data_dimension + k
                        rows[index] = i
                        cols[index] = connectivity_idx[i, j] * data_dimension + k
                        data[index] = weights[i, j * data_dimension + k]
            
            cdef:
                sp.csr_matrix weights_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_target, n_elems))
            return weights_sparse

    cdef tuple prepare_interpolator(self, str method, 
                                   const DTYPE_F_t[::1] source_variable, const int data_dimension, 
                                   const DTYPE_I_t[::1] target_points):
        
        
        cdef int n_target = len(target_points)
        cdef int n_columns = self.grid_obj.MX_ELEMENTS_PER_POINT * data_dimension
        cdef int dim = self.grid_obj.n_dims

        cdef DTYPE_F_t[:, ::1] connectivity_val  = np.zeros((n_target, n_columns), dtype=DTYPE_F)
        cdef DTYPE_I_t[:, ::1] connectivity_idx  = np.ones((n_target, self.grid_obj.MX_ELEMENTS_PER_POINT), dtype=DTYPE_I) * -1
        cdef DTYPE_F_t[:, ::1] weights           = np.zeros((n_target, n_columns), dtype=DTYPE_F)

        cdef DTYPE_F_t[:, ::1] target_coordinates = np.asarray(self.grid_obj.point_coords)[target_points]
        cdef DTYPE_F_t[:, ::1] source_coordinates = self.grid_obj.centroids 

        cdef int point, elem, first, last, i, j, k

        # Populate connectivity_val
        cdef int use_threads = min(8, np.ceil(n_target / 800))

        omp_set_num_threads(use_threads)
        for i in prange(n_target, nogil=True, schedule='static', num_threads=use_threads):
            point = target_points[i]
            first = self.grid_obj.esup_ptr[point]
            last = self.grid_obj.esup_ptr[point + 1]
            for j, elem in enumerate(self.grid_obj.esup[first:last]):
                connectivity_idx[i, j] = elem
                for k in range(data_dimension):
                    connectivity_val[i, j * data_dimension + k] = source_variable[elem * data_dimension + k]
        
        if method == "inv_dist":
            
            
            self.supported_methods[method](
                dim,
                target_coordinates, source_coordinates, 
                data_dimension,
                connectivity_idx, weights   
            )
        
        return weights, connectivity_idx, connectivity_val


