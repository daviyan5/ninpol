"""
This file contains the "Interpolator" class definition, for interpolating mesh unknows. 
"""
import numpy as np
import meshio

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
            "inv_dist": self.inv_dist_interpolator
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

        self.grid_obj = Grid(args[0], 
                             args[1], args[2], args[3], 
                             args[4], args[5], args[6], 
                             args[7], args[8])

        self.grid_obj.build(args[9], args[10], args[11])
        
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
            int n_elems = 0, n_faces = 0

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
            nfael_e = len(elem_type[faces_key] if faces_key in elem_type else [])
            lnofa_e = [len(face) for face in elem_type[faces_key]] if faces_key in elem_type else []
            lpofa_e = elem_type[faces_key] if faces_key in elem_type else []

            npoed_e = len(elem_type["edges"] if "edges" in elem_type else [])
            lpoed_e = elem_type["edges"] if "edges" in elem_type else []

            type_index = elem_type["element_type"]
            
            nfael[type_index] = nfael_e
            if faces_key in elem_type:
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
            if faces_key in self.point_ordering["elements"][elem_type_str]:   
                n_faces += len(CellBlock.data) * len(self.point_ordering["elements"][elem_type_str][faces_key])
        

        cdef: 
            dict point_tag_to_index = {}
            dict cell_tag_to_index = {}

            int point_index = 0
            int cell_index = 0    

            int elem_type_index
            
            DTYPE_I_t[:, ::1] connectivity   = np.ones((n_elems, NINPOL_MAX_POINTS_PER_ELEMENT), dtype=DTYPE_I) * -1
            DTYPE_I_t[::1] element_types     = np.ones(n_elems, dtype=DTYPE_I) * -1
            DTYPE_I_t[::1] n_points_per_elem = np.ones(n_elems, dtype=DTYPE_I) * -1
        
        for CellBlock in mesh.cells:
            elem_type_str = CellBlock.type
            elem_type_index = self.point_ordering["elements"][elem_type_str]["element_type"]

            for i, cell in enumerate(CellBlock.data):
                for j, point in enumerate(cell):
                    connectivity[cell_index, j] = point
                element_types[cell_index] = elem_type_index
                n_points_per_elem[cell_index] = len(cell)
                cell_index += 1

        return (dim, n_elems, n_points, n_faces, 
                nfael, lnofa, lpofa, 
                nedel, lpoed,
                connectivity, element_types, n_points_per_elem)

    
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

    def interpolate(self, DTYPE_I_t[::1] target, str variable,  str method, str itype = "ctp"):
        # Interpolates a 'variable' defined in the source (cells, if itype is "ctp", or points, if itype is "ctp") to the
        # target (points, if itype is "ctp", or cells, if itype is "ptc") using the specified method.
        if not self.is_grid_initialized:
            raise ValueError("Grid not initialized. Please load a mesh first.")
        
        if method not in self.supported_methods:
            raise ValueError(f"Method '{method}' not supported. Supported methods are: {list(self.supported_methods.keys())}")

        if itype not in ["ctp", "ptc"]:
            raise ValueError(f"Invalid itype '{itype}'. Supported itypes are 'ptc' and 'ctp'.")
        
        if variable not in (self.variable_to_index["cells" if itype == "ctp" else "points"]):
            raise ValueError(f"Variable '{variable}' not found in {itype} data.")

        cdef:
            int data_index = 0
            int data_dimension = 1
            DTYPE_F_t[::1] source_data = np.zeros(1, dtype=DTYPE_F)
            str source_type = "cells" if itype == "ctp" else "points"

        if itype == "ctp":
            if variable not in self.variable_to_index["cells"]:
                raise ValueError(f"Variable '{variable}' not found in cells data.")

            data_index     = self.variable_to_index["cells"][variable]
            source_data    = self.cells_data[data_index]
            data_dimension = self.cells_data_dimensions[data_index]

        if itype == "ptc":
            if variable not in self.variable_to_index["points"]:
                raise ValueError(f"Variable '{variable}' not found in points data.")

            data_index     = self.variable_to_index["points"][variable]
            source_data    = self.points_data[data_index]
            data_dimension = self.points_data_dimensions[data_index]
        
        return np.asarray(self.supported_methods[method](source_data, data_dimension, source_type, target))

    cdef DTYPE_F_t[::1] inv_dist_interpolator(self, const DTYPE_F_t[::1] source_data, DTYPE_I_t data_dimension, 
                                            str source_type, const DTYPE_I_t[::1] target):
        
        cdef dim = self.grid_obj.n_dims

        cdef:
            DTYPE_F_t[::1] interpolated_data = np.zeros(len(target) * data_dimension, dtype=DTYPE_F)

        if source_type == "cells":
            interpolated_data = distance_inverse(dim, target           = np.asarray(self.grid_obj.point_coords)[target], 
                                                      source           = self.grid_obj.centroids,
                                                      connectivity     = self.grid_obj.esup,
                                                      connectivity_ptr = self.grid_obj.esup_ptr,
                                                      weights_shape    = data_dimension,
                                                      weights          = source_data
                                                )
        return interpolated_data

