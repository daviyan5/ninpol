"""
This file contains the "Grid" class implementation
"""
import numpy as np

from cython.parallel import prange
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num


cdef class Grid:
    def __cinit__(self, DTYPE_I_t n_dims, 
                        DTYPE_I_t n_elems, DTYPE_I_t n_points, DTYPE_I_t n_faces, 
                        DTYPE_I_t[::1] nfael, DTYPE_I_t[:, ::1] lnofa, DTYPE_I_t[:, :, ::1] lpofa, 
                        DTYPE_I_t[::1] nedel, DTYPE_I_t[:, :, ::1] lpoed):
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
                n_faces : int
                    Number of faces in the mesh
                nfael : array_like
                    Number of faces per element
                lnofa : array_like
                    Number of points per face
                lpofa : array_like
                    Points per face
                
        """
        self.n_dims   = n_dims

        self.n_elems  = n_elems
        self.n_faces  = n_faces
        self.n_points = n_points

        self.nfael    = nfael.copy()
        self.lnofa    = lnofa.copy()
        self.lpofa    = lpofa.copy()

        self.nedel    = nedel.copy()
        self.lpoed    = lpoed.copy()

        self.are_elements_loaded = False
        self.are_coords_loaded = False
        


    cpdef void build(self, const DTYPE_I_t[:, ::1] connectivity, const DTYPE_I_t[::1] element_types, const DTYPE_I_t[::1] n_points_per_elem):

        # Check that the connectivity matrix is not None and has the correct shape
        if connectivity is None:
            raise ValueError("The connectivity matrix cannot be None.")
        if connectivity.shape[0] != self.n_elems:
            raise ValueError("The number of rows in the connectivity matrix must be equal to the number of elements.")
        
        self.inpoel = connectivity.copy()
        self.element_types = element_types.copy()
        self.n_points_per_elem = n_points_per_elem.copy()
        
        self.are_elements_loaded = True

        # Calculate number of points per element, defines number of faces per element
        cdef:
            int i, j
        
        # Calculate the elements surrounding each point
        self.build_esup()

        # Calculate the points surrounding each point
        self.build_psup()

        # Calculate the faces composing each element
        self.build_infael()
        
        # Calculate the elements surrounding each face
        self.build_esufa()

        # Calculate the elements surrounding each element
        self.build_esuel()

        # Calculate the edges surrounding each element
        self.build_inedel()

    cdef void build_esup(self):
        
        cdef:
            int i, j

        # Reshape the arrays
        self.esup_ptr = np.zeros(self.n_points+1, dtype=np.int64)

        # Count the number of elements surrounding each point    
        for i in range(self.n_elems):
            for j in range(self.n_points_per_elem[i]):
                self.esup_ptr[self.inpoel[i, j] + 1] += 1
        
        # Compute the cumulative sum of the number of elements surrounding each point
        for i in range(self.n_points):
            self.esup_ptr[i + 1] += self.esup_ptr[i]

        # Fill the esup array
        self.esup = np.zeros(self.esup_ptr[self.n_points], dtype=np.int64)
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
            DTYPE_I_t[::1] temp_psup = np.ones(self.n_points, dtype=np.int64) * -1
        
        self.psup_ptr = np.zeros(self.n_points+1, dtype=np.int64)
        self.psup_ptr[0] = 0
    
        # Upper bound for the number of points surrounding each point, pehaps this can be improved
        self.psup = np.zeros((self.esup_ptr[self.n_points] * (7)), dtype=np.int64) 

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
    
    cdef void build_infael(self):
        pass
    
    cdef void build_esufa(self):
        pass

    cdef void build_esuel(self):
        
        # Declare every variable
        cdef:
            int j, k, l, m
            int ielem, jelem
            int ielem_type, jelem_type

            DTYPE_I_t[::1] ielem_face = np.zeros(4, dtype=np.int64)
            DTYPE_I_t[::1] jelem_face = np.zeros(4, dtype=np.int64)

            DTYPE_I_t[::1] ielem_face_index = np.zeros(4, dtype=np.int64)
            DTYPE_I_t[::1] jelem_face_index = np.zeros(4, dtype=np.int64)

            int point, kpoint
            int num_elems, num_elems_min
            int found_elem
            int jelem_face_point
            int is_equal 
            

        self.esuel = np.ones((self.n_elems, 6), dtype=np.int64) * -1

        # For each element
        for ielem in range(self.n_elems):
            ielem_type = self.element_types[ielem]

            # For each face
            for j in range(self.nfael[ielem_type]):

                if self.esuel[ielem, j] != -1:
                    continue
                    
                # Choose a point from the face
                ielem_face_index = self.lpofa[ielem_type, j]
                
                for k in range(self.lnofa[ielem_type, j]):
                    ielem_face[k] = self.inpoel[ielem, ielem_face_index[k]]
                
                for k in range(4 - self.lnofa[ielem_type, j]):
                    ielem_face[3 - k] = -1

                point = ielem_face[0]
                num_elems_min = self.esup_ptr[point+1] - self.esup_ptr[point]

                # Choose the point with the least number of elements around it
                for k in range(self.lnofa[ielem_type, j]):
                    kpoint = ielem_face[k]
                    num_elems = self.esup_ptr[kpoint+1] - self.esup_ptr[kpoint]
                    if num_elems < num_elems_min:
                        point = kpoint
                        num_elems_min = num_elems


                found_elem = False

                # For each element around the point
                for k in range(self.esup_ptr[point], self.esup_ptr[point+1]):
                    jelem = self.esup[k]
                    jelem_type = self.element_types[jelem]

                    # If the element around the point is not the current element
                    if jelem != ielem:

                        # For each face of the element around the point
                        for l in range(self.nfael[jelem_type]):
                            jelem_face_index = self.lpofa[jelem_type, l]
                            is_equal = 0
                            for m in range(self.lnofa[jelem_type, l]):
                                jelem_face_point = self.inpoel[jelem, jelem_face_index[m]]

                                if(jelem_face_point == ielem_face[0] or 
                                   jelem_face_point == ielem_face[1] or 
                                   jelem_face_point == ielem_face[2] or 
                                   jelem_face_point == ielem_face[3]):
                                    is_equal += 1
                        
                            if is_equal == self.lnofa[ielem_type, j]:

                                # Add the element around the point to the list of elements around the current element
                                self.esuel[ielem, j] = jelem
                                # Add the current element to the list of elements around the element around the point
                                self.esuel[jelem, l] = ielem

                                found_elem = True

                            if found_elem:
                                break
                                
                    if found_elem:
                        break
    
    cdef void build_inedel(self):
        pass
    
    cdef void load_point_coords(self, DTYPE_F_t[:, ::1] coords):
        self.point_coords = coords.copy()
        self.are_coords_loaded = True

    cdef void calculate_cells_centroids(self):
        if self.are_elements_loaded == False:
            raise ValueError("The element types have not been set.")
        if self.are_coords_loaded == False:
            raise ValueError("The point coordinates have not been set.")

        # For each element type, calculate the centroid
        # OBS: The right way to do this is to:
        #           1. Triangulate each face
        #           2. Calculate the contribuitions of each face to the volume and centroid
        # This can be done by using an array 'triangles' for each face, such that triangles[i] = [[a1, b1, c1], [a2, b2, c2], ...]
        # that triangulate that face. If the face is already a triangle, then triangles[i] = [[a, b, c]]
        # Then, using esufa, we can calculate the volume and centroid of each element

        # However, for now, we will use only the average of the points
        cdef:
            int i, j, k

        self.centroids = np.zeros((self.n_elems, self.n_dims), dtype=np.float64)
        
        cdef:
            int use_threads = self.n_elems > 1000
            float machine_epsilon = 10 ** int(np.log10(np.finfo(np.float64).eps))

        omp_set_num_threads(8 if use_threads else 1)

        for i in prange(self.n_elems, nogil=True, schedule='static', num_threads=8 if use_threads else 1):
            for j in range(self.n_points_per_elem[i]):
                for k in range(self.n_dims):
                    self.centroids[i, k] += self.point_coords[self.inpoel[i, j], k] / self.n_points_per_elem[i]

        