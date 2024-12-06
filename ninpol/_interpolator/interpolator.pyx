"""
This file contains the "Interpolator" class definition, for interpolating mesh unknows. 
"""
import numpy as np
import scipy.sparse as sp

import time
import meshio
import pickle
import tempfile
import os

from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num
from cython.parallel import prange

cdef extern from *:
    """
    #ifdef _WIN32
    #ifdef MS_WINDOWS
    #define CLOCK_REALTIME 0
    static int clock_gettime(int clk_id, struct timespec *tp) {
        tp->tv_sec = 0;
        tp->tv_nsec = 0;
        return 0;
    }
    #endif
    #endif
    """
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

DTYPE_I = np.int64
DTYPE_F = np.float64


cdef class Interpolator:

    def __cinit__(self, str name = "interpolator", int logging = False, int build_edges = False):
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
        self.build_edges = build_edges

        self.gls = GLSInterpolation(logging)

        self.idw = IDWInterpolation(logging)

        self.ls  = LSInterpolation(logging)

        self.supported_methods = {
            "gls": self.gls.prepare,
            "idw": self.idw.prepare,
            "ls":  self.ls.prepare
        }

        self.variable_to_index = {
            "points": {},
            "cells": {},
            "faces": {}
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

        self.faces_data = np.zeros((1, 1), dtype=DTYPE_F)
        self.faces_data_dimensions = np.zeros(1, dtype=DTYPE_I)

        self.logging = logging
        self.logger  = Logger(name, logging=logging)

        
        self.CACHE_PATH = tempfile.gettempdir()

    def is_cached(self, str filename):
        if filename == "":
            return None
        cdef:
            str final_path
            str cache_name
            int last_index
            str little_hash 
            str divisor
        
        # hash the size of the file to hexa
        little_hash = hex(os.path.getsize(filename))
        divisor = os.path.sep
        last_index = len(filename.split(divisor)) - 1
        cache_name = filename.split(divisor)[last_index].split(".")[0] + little_hash + ".pkl"
        final_path = os.path.join(self.CACHE_PATH, cache_name)
        if os.path.exists(final_path):
            return final_path
        return None

    cdef dict make_cache(self, tuple args):
        cdef:
            dict cache
            dict grid_cache
            dict int_cache
            tuple new_args
        
        new_args = (
            args[0], args[1], args[2],
            np.asarray(args[3]), 
            np.asarray(args[4]), np.asarray(args[5]), np.asarray(args[6]),
            np.asarray(args[7]), np.asarray(args[8]),
            np.asarray(args[9]), np.asarray(args[10]),
            args[11], args[12]
        )
        
        int_cache = {
            "cells_data": np.asarray(self.cells_data),
            "cells_data_dimensions": np.asarray(self.cells_data_dimensions),

            "points_data": np.asarray(self.points_data),
            "points_data_dimensions": np.asarray(self.points_data_dimensions),

            "faces_data": np.asarray(self.faces_data),
            "faces_data_dimensions": np.asarray(self.faces_data_dimensions),

            "variable_to_index": self.variable_to_index,
            "points_coords": np.asarray(self.points_coords)
        }
        cache = {
            "grid": new_args,
            "interpolator": int_cache
        }
        return cache

    cdef void load_cache(self, dict cache):
        cdef:
            dict int_cache
            
        self.grid = Grid(*cache["grid"])

        int_cache = cache["interpolator"]
        self.cells_data = int_cache["cells_data"]
        self.cells_data_dimensions = int_cache["cells_data_dimensions"]

        self.points_data = int_cache["points_data"]
        self.points_data_dimensions = int_cache["points_data_dimensions"]

        self.faces_data = int_cache["faces_data"]
        self.faces_data_dimensions = int_cache["faces_data_dimensions"]

        self.variable_to_index = int_cache["variable_to_index"]
        self.points_coords = int_cache["points_coords"]
        
    
    cpdef void load_mesh(self, str filename = "", object mesh_obj = None):
        if filename == "" and mesh_obj is None:
            raise ValueError("Filename for the mesh or meshio.Mesh object must be provided.")
        # Loads a mesh from a file

        cdef:
            object f
            dict cache
        
        cdef:
            tuple args
        
        if self.is_cached(filename):
            self.logger.log(f"Loading mesh from cache", "INFO")
            with open(self.is_cached(filename), "rb") as f:
                cache = pickle.load(f)
                self.load_cache(cache)
        else:
            if filename != "":
                self.logger.log(f"Reading mesh from {filename}", "INFO")
                self.mesh_obj = meshio.read(filename)
            else:
                self.logger.log(f"Using mesh object", "INFO")
                self.mesh_obj = mesh_obj

            args = self.process_mesh(self.mesh_obj)
            self.grid = Grid(*args)
            self.points_coords = self.mesh_obj.points.astype(DTYPE_F)
            
        cdef:
            double start_time = 0., end_time = 0.
            timespec ts
        
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + (ts.tv_nsec / 1e9)

        self.grid.build()
        self.grid.load_point_coords(self.points_coords)
        self.grid.calculate_centroids()
        self.grid.calculate_normal_faces()
        
        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + (ts.tv_nsec / 1e9)

        self.logger.log(f"Grid built in {end_time - start_time:.2f} seconds", "INFO") 
        
        if not self.is_cached(filename):
            clock_gettime(CLOCK_REALTIME, &ts)
            start_time = ts.tv_sec + (ts.tv_nsec / 1e9)
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
            clock_gettime(CLOCK_REALTIME, &ts)
            end_time = ts.tv_sec + (ts.tv_nsec / 1e9)

            self.logger.log(f"Data loaded in {end_time - start_time:.2f} seconds", "INFO")

        self.is_grid_initialized = True

        cdef:
            int last_index
            str pkl_name
            str final_path
            str little_hash
            str divisor

        self.logger.log(f"Mesh loaded successfully: {self.grid.n_points} points and {self.grid.n_elems} elements.", "INFO")
        
        if not self.is_cached(filename) and filename != "":
            little_hash = hex(os.path.getsize(filename))
            divisor = os.path.sep
            last_index = len(filename.split(divisor)) - 1
            pkl_name = filename.split(divisor)[last_index].split(".")[0] + little_hash + ".pkl"
            final_path = os.path.join(self.CACHE_PATH, pkl_name)
            with open(final_path, "wb") as f:
                pickle.dump(self.make_cache(args), f)
            self.logger.log(f"Caching grid to {final_path}", "INFO")
            
        
    cdef tuple process_mesh(self, object mesh):
        cdef:
            int dim = 1
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

        # Get the dimension of the mesh
        for key in mesh.cells:
            for dimension in self.types_per_dimension:
                if key.type in self.types_per_dimension[dimension]:
                    dim = max(dim, dimension)
        
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
                connectivity, element_types,
                self.logging, self.build_edges)

    
    cdef void load_data(self, dict data_dict, str data_type):
    # Data must be homogeneous
    # Only supports scalar and vector data. Each vector must have the same dimension

        cdef:
            int n_variables = len(data_dict)
            int n_elements = self.grid.n_elems if data_type == "cells" else self.grid.n_points
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

            self.logger.log(f"Loading {data_type} data for variable '{variable}'", "INFO")
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

    cdef void load_cell_data(self):
        # cell_data_dict will save each variable separately for element type. We dont want that.
        
        cdef:
            int dim = self.grid.dim
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

    cdef void load_point_data(self):
        self.load_data(self.mesh_obj.point_data, "points")

    cpdef void load_face_data(self, dict data_dict, DTYPE_I_t[:, ::1] face_connectivity = np.array([[]], dtype=int)):
        """
        Since meshio can't properly provide face data in all cases, this function shall be used
        to load face data from a dictionary. The face_connectivity array can be provided if the user
        believes the connectivity (inpofa) from self.grid is different from it's own. 
        
        Data_dict should be a dictionary with the following structure:
        {
            "variable_name": np.array([data])
        }

        Vector data is not supported yet.
        """
        
        # If face_connectivity is not empty, let's create a mapping from the face_connectivity to the grid connectivity
        cdef:
            cnp.ndarray[int, ndim=2, mode="c"] A = face_connectivity
            cnp.ndarray[int, ndim=2, mode="c"] B = self.grid.inpofa
            DTYPE_I_t[::1] face_to_grid = np.arange(self.grid.n_faces, dtype=DTYPE_I)
            
        if len(face_connectivity) > 0:
            # Create views of A and B where each row is represented as a single item
            A_view = A.view([('', A.dtype)] * A.shape[1]).ravel()
            B_view = B.view([('', B.dtype)] * B.shape[1]).ravel()
            
            # Sort B_view and get the sorted indices
            idx_B_sorted = np.argsort(B_view)
            B_view_sorted = B_view[idx_B_sorted]
            
            # Use searchsorted to find the indices of A_view in the sorted B_view
            idx_in_B_sorted = np.searchsorted(B_view_sorted, A_view)
            
            # Map the indices back to the original indices in B
            face_to_grid = idx_B_sorted[idx_in_B_sorted]
        
        cdef:
            int i
            str variable

        self.faces_data = np.zeros((len(data_dict), self.grid.n_faces), dtype=DTYPE_F)
        for i, variable in enumerate(data_dict):
            self.variable_to_index["faces"][variable] = i
            self.faces_data_dimensions[i] = data_dict[variable].shape[1]
            self.faces_data[i] = data_dict[variable][face_to_grid].astype(DTYPE_F)

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
    
    cpdef tuple interpolate(self, str variable, str method, DTYPE_I_t[::1] target_points = np.array([], dtype=DTYPE_I)):
        
        if not self.is_grid_initialized:
            raise ValueError("Grid not initialized. Please load a mesh first.")
        
        if method not in self.supported_methods:
            raise ValueError(f"Method '{method}' not supported. Supported methods are: {list(self.supported_methods.keys())}")

        if len(target_points) == 0:
            target_points = np.arange(self.grid.n_points, dtype=DTYPE_I)

        cdef:
            int data_index = 0
            int data_dimension = 1

        if variable not in self.variable_to_index["cells"]:
            raise ValueError(f"Variable '{variable}' not found in cells data. Point -> Cell interpolation not supported yet.")

        data_index     = self.variable_to_index["cells"][variable]
        data_dimension = self.cells_data_dimensions[data_index]
        
        if data_dimension > 1:
            raise ValueError(f"Variable '{variable}' has more than one dimension. Vector data not supported yet.")

        cdef: 
            DTYPE_F_t[:, ::1] weights 
            DTYPE_F_t[::1] neumann_ws

        self.logger.log(f"Interpolating variable '{variable}' using method '{method}'", "INFO")

        weights, neumann_ws = self.prepare_interpolator(method, variable, target_points)
        
        cdef:
            int i, j
            int n_target    = len(target_points)
            int n_elems     = self.grid.n_elems
            int data_size   = self.grid.esup.shape[0]

            int use_threads = min(16, np.ceil(n_target / 400))

            
            
        self.logger.log(f"Interpolation prepared successfully", "INFO")
        
        
        cdef:
            int point
            float start_time = 0., end_time = 0.
            timespec ts
            DTYPE_I_t[::1] rows = np.ones(data_size, dtype=DTYPE_I) * -1
            DTYPE_I_t[::1] cols = np.ones(data_size, dtype=DTYPE_I) * -1
            DTYPE_F_t[::1] data = np.ones(data_size, dtype=DTYPE_F) * -1    

        # Convert weights from (n_target, n_columns) to (n_target, n_elems) using sparse matrix
        index = 0
        

        
        self.logger.log(f"Building sparse matrix with {data_size} non-zero elements", "INFO")
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + (ts.tv_nsec / 1e9)

        #for i in prange(n_target, nogil=True, schedule='static', num_threads=use_threads):
        for i in range(n_target):
            point   = target_points[i]
            for j in range(self.grid.esup_ptr[point], self.grid.esup_ptr[point + 1]):
                
                rows[j] = point
                cols[j] = self.grid.esup[j]
                data[j] = weights[i, j - self.grid.esup_ptr[point]] + neumann_ws[i]
        
        cdef:
            object weights_sparse
        weights_sparse = sp.csr_matrix((data, (rows, cols)), 
                                        shape=(n_target, n_elems))
        weights_sparse.eliminate_zeros()
        
        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + (ts.tv_nsec / 1e9)
        self.logger.log(f"Returning Weights matrix ({end_time - start_time:.2f})", "INFO")
        return weights_sparse, np.asarray(neumann_ws)

    cdef tuple prepare_interpolator(self, str method, str variable,
                                    const DTYPE_I_t[::1] target_points):
        """
        Every interpolation method shall be called with:
            - The Grid
            - The Target Points
            - Cell Data
            - Point Data
            - Weights Matrix  (out)
            - Neumann Weights (out)

        """
        cdef:
            int n_target = len(target_points)
            int n_columns = self.grid.MX_ELEMENTS_PER_POINT

            double start_time = 0., end_time = 0.
            timespec ts

            DTYPE_F_t[:, ::1] weights  = np.zeros((n_target, n_columns), dtype=DTYPE_F)
            DTYPE_F_t[::1] neumann_ws  = np.zeros(n_target, dtype=DTYPE_F)
        
        self.logger.log(f"Preparing interpolator for method '{method}'", "INFO")
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + (ts.tv_nsec / 1e9)

        self.supported_methods[method](
            self.grid, 
            self.cells_data, self.points_data, self.faces_data,
            self.variable_to_index, 
            variable,
            target_points, 
            weights, neumann_ws
        )

        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + (ts.tv_nsec / 1e9)
        self.logger.log(f"Interpolation done in {end_time - start_time:.2f} seconds", "INFO")

        return weights, neumann_ws