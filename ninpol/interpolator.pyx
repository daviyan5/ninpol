"""
This file contains the "Interpolator" class definition, for interpolating mesh unknows. 
"""
import numpy as np
import meshio
from .mesh cimport grid

DTYPE_I = np.int64
DTYPE_F = np.float64

cdef type MeshioMesh = meshio._mesh.Mesh

cdef class Interpolator:

    def __cinit__(self):
        # Load point-ordering.yaml 
        import yaml
        import os

        cdef:
            str current_directory = os.path.dirname(os.path.abspath(__file__))
            str point_ordering_path = os.path.join(current_directory, "utils", "point_ordering.yaml")
        
        
        with open(point_ordering_path, 'r') as f:
            self.point_ordering = yaml.load(f, Loader=yaml.FullLoader)

        self.is_grid_initialized = False
    
    def load_mesh(self, str filename):
        # Loads a mesh from a file
        self.mesh_obj = meshio.read(filename)
        cdef tuple args = self.process_mesh(self.mesh_obj)

        self.grid_obj = grid.Grid(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        self.grid_obj.build(args[7], args[8], args[9])
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

            str elem_type_str
            dict elem_type
            int type_index, i, j

            DTYPE_I_t[::1] nfael       = np.ones(8, dtype=DTYPE_I) * -1
            DTYPE_I_t[:, ::1] lnofa    = np.ones((8, 6), dtype=DTYPE_I) * -1
            DTYPE_I_t[:, :, ::1] lpofa = np.ones((8, 6, 4), dtype=DTYPE_I) * -1

        for elem_type in self.point_ordering["elements"].values():
            nfael_e = len(elem_type["faces"] if "faces" in elem_type else [])
            lnofa_e = [len(face) for face in elem_type["faces"]] if "faces" in elem_type else []
            lpofa_e = elem_type["faces"] if "faces" in elem_type else []

            type_index = elem_type["element_type"]
            
            nfael[type_index] = nfael_e
            if "faces" in elem_type:
                for i, n_point in enumerate(lnofa_e):
                    lnofa[type_index, i] = n_point
                    
                for i, face in enumerate(lpofa_e):
                    for j, point in enumerate(face):
                        lpofa[type_index, i, j] = point


        for CellBlock in mesh.cells:
            elem_type_str = CellBlock.type
            
            n_elems += len(CellBlock.data)
            if "faces" in self.point_ordering["elements"][elem_type_str]:   
                n_faces += len(CellBlock.data) * len(self.point_ordering["elements"][elem_type_str]["faces"])
        

        cdef: 
            dict point_tag_to_index = {}
            dict cell_tag_to_index = {}

            int point_index = 0
            int cell_index = 0    

            int elem_type_index
            
            DTYPE_I_t[:, ::1] connectivity   = np.ones((n_elems, 8), dtype=DTYPE_I) * -1
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
                connectivity, element_types, n_points_per_elem)
        
