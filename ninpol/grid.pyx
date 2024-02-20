"""
This file contains the "Grid" class implementation
"""

import numpy as np
from libc.stdio cimport printf
from cython.parallel cimport parallel, prange
cimport openmp


DTYPE_I = np.int64
DTYPE_F = np.float64

cdef class Grid:
    def __cinit__(self, DTYPE_I_t n_dims, DTYPE_I_t n_elems, DTYPE_I_t n_points):
        """
            Initializes the grid.

            Parameters
            ---------
                n_dims : int
                    Number of dimensions (i.e. 2 or 3)
                n_elems : int
                    Number of elements in the mesh
                n_points : int
                    Number of points (vertices) in the mesh
        """
        import yaml
        import os
        cdef:
            str script_dir = os.path.dirname(os.path.abspath(__file__))
            str point_ordering_path = os.path.join(script_dir, "utils/point_ordering.yaml")
            dict point_ordering_obj, point_ordering

        self.n_dims             = n_dims
        self.n_elems            = n_elems
        self.n_points           = n_points
        point_ordering_obj      = yaml.load(open(point_ordering_path), Loader=yaml.FullLoader)
        point_ordering          = point_ordering_obj['elements']

        cdef:
            DTYPE_I_t[::1] nfael = np.zeros(16, dtype=DTYPE_I)
            DTYPE_I_t[:, ::1] lnofa = np.zeros((16, 6), dtype=DTYPE_I)
            DTYPE_I_t[:, :, ::1] lpofa = np.ones((16, 6, 4), dtype=DTYPE_I) * -1
            int element_type, element_n_points
            int i, j
            list face
            int point
            str element_name
            dict element
        
        self.n_point_to_type = np.zeros(16, dtype=DTYPE_I)

        for element_name in point_ordering:
            element = point_ordering[element_name]
            element_type = element['element_type']
            element_n_points = element['number_of_points']
            self.n_point_to_type[element_n_points] = element_type

            if element_n_points < 3:
                continue

            nfael[element_type] = len(element['faces'])
            for i, face in enumerate(element['faces']):
                lnofa[element_type, i] = len(face)
                for j, point in enumerate(face):
                    lpofa[element_type, i, j] = point
                
                if len(face) < 4:
                    lpofa[element_type, i, 3] = -1
        
        self.nfael = nfael
        self.lnofa = lnofa
        self.lpofa = lpofa


    cpdef void build(self, DTYPE_I_t[:, ::1] connectivity, DTYPE_I_t[::1] element_types = None):

        # Check that the connectivity matrix is not None and has the correct shape
        if connectivity is None:
            raise ValueError("The connectivity matrix cannot be None.")
        if connectivity.shape[0] != self.n_elems:

            raise ValueError("The number of rows in the connectivity matrix must be equal to the number of elements.")
        
        self.inpoel = connectivity.copy()
        
        # Calculate number of points per element, defines number of faces per element
        cdef:
            int i, j

        self.n_points_per_elem = np.zeros(self.n_elems, dtype=DTYPE_I)
        self.element_types = element_types if element_types is not None else np.zeros(self.n_elems, dtype=DTYPE_I)

        for i in range(self.n_elems):
            for j in range(self.inpoel.shape[1]):
                if self.inpoel[i, j] == -1:
                    break
                self.n_points_per_elem[i] += 1
            if element_types is None:
                self.element_types[i] = self.n_point_to_type[self.n_points_per_elem[i]]
        
        # Calculate the elements surrounding each point
        self.build_esup()

        # Calculate the points surrounding each point
        self.build_psup()

        # Calculate the elements surrounding each element
        # self.build_esuel()

        # Calculate the points that form each edge
        if self.n_dims == 3:
            self.build_inpoed()
            self.build_ledel()

    cdef void build_esup(self):
        
        cdef:
            int i, j

        # Reshape the arrays
        self.esup_ptr = np.zeros(self.n_points+1, dtype=DTYPE_I)

        # Count the number of elements surrounding each point    
        for i in range(self.n_elems):
            for j in range(self.n_points_per_elem[i]):
                self.esup_ptr[self.inpoel[i, j] + 1] += 1
        
        # Compute the cumulative sum of the number of elements surrounding each point
        for i in range(self.n_points):
            self.esup_ptr[i + 1] += self.esup_ptr[i]

        # Fill the esup array
        self.esup = np.zeros(self.esup_ptr[self.n_points], dtype=DTYPE_I)
        for i in range(self.n_elems):
            for j in range(self.n_points_per_elem[i]):
                self.esup[self.esup_ptr[self.inpoel[i, j]]] = i
                self.esup_ptr[self.inpoel[i, j]] += 1
        
        for i in range(self.n_points, 0, -1):
            self.esup_ptr[i] = self.esup_ptr[i-1]
        self.esup_ptr[0] = 0

    cdef void build_psup(self):

        cdef:
            int i, j, k
            int stor_ptr = 0, point_idx
            DTYPE_I_t[::1] temp_psup = np.ones(self.n_points, dtype=DTYPE_I) * -1
        
        self.psup_ptr = np.zeros(self.n_points+1, dtype=DTYPE_I)
        self.psup_ptr[0] = 0
    
        # Upper bound for the number of points surrounding each point, pehaps this can be improved
        self.psup = np.zeros((self.esup_ptr[self.n_points] * (7)), dtype=DTYPE_I) 

        # Calculate the points surrounding each point, using temp_psup to avoid duplicates
        for i in range(self.n_points):
            for j in range(self.esup_ptr[i], self.esup_ptr[i+1]):
                for k in range(self.n_points_per_elem[self.esup[j]]):
                    point_idx = self.inpoel[self.esup[j], k]
                    if point_idx != i and temp_psup[point_idx] != i:
                        self.psup[stor_ptr] = point_idx
                        temp_psup[point_idx] = i
                        stor_ptr += 1
                        
            self.psup_ptr[i+1] = stor_ptr

        # Resize the psup array to remove padding
        self.psup = self.psup[:stor_ptr]

    cdef void build_esuel(self):

        # Declare every variable
        cdef:
            int j, k, l, m
            int ielem, jelem
            int ielem_type, jelem_type
            DTYPE_I_t[::1] ielem_face = np.zeros(4, dtype=DTYPE_I)
            DTYPE_I_t[::1] jelem_face = np.zeros(4, dtype=DTYPE_I)
            int point, kpoint
            int num_elems, num_elems_min
            int found_elem

        self.esuel = np.ones((self.n_elems, 6), dtype=DTYPE_I) * -1
        # For each element
        for ielem in range(self.n_elems):
            ielem_type = self.element_types[ielem]

            # For each face
            for j in range(self.nfael[ielem_type]):
                # Choose a point from the face
                ielem_face = self.lpofa[ielem_type, j].copy()
                for k in range(self.lnofa[ielem_type, j]):
                    ielem_face[k] = self.inpoel[ielem, ielem_face[k]]

                point = ielem_face[0]
                num_elems_min = self.esup_ptr[point+1] - self.esup_ptr[point]

                # Choose the point with the least number of elements around it
                for k in range(self.lnofa[ielem_type, j]):
                    kpoint = ielem_face[k]
                    num_elems = self.esup_ptr[kpoint+1] - self.esup_ptr[kpoint]
                    if num_elems < num_elems_min:
                        point = kpoint
                        num_elems_min = num_elems

                ielem_face = np.sort(ielem_face)
                found_elem = False

                # For each element around the point
                for k in range(self.esup_ptr[point], self.esup_ptr[point+1]):
                    jelem = self.esup[k]
                    jelem_type = self.element_types[jelem]

                    # If the element around the point is not the current element
                    if jelem != ielem:

                        # For each face of the element around the point
                        for l in range(self.nfael[jelem_type]):
                            
                            jelem_face = self.lpofa[jelem_type, l].copy()

                            for m in range(self.lnofa[jelem_type, l]):
                                jelem_face[m] = self.inpoel[jelem, jelem_face[m]]
                                
                            jelem_face = np.sort(jelem_face)

                            # If the face of the element around the point is equal to the face of the current element
                            if np.array_equal(ielem_face, jelem_face):

                                # Add the element around the point to the list of elements around the current element
                                self.esuel[ielem, j] = jelem
                                # Add the current element to the list of elements around the element around the point
                                self.esuel[jelem, l] = ielem

                                found_elem = True
                                break

                        if found_elem:
                            break
                
               
    cdef void build_inpoed(self):
        pass
    cdef void build_ledel(self):
        pass
    

        