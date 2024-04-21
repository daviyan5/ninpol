"""
This file contains the "Grid" class implementation
"""

import numpy as np

from cython.parallel import prange

cdef:
    type DTYPE_I = np.int64
    type DTYPE_F = np.float64

cdef class Grid:
    def __cinit__(self, DTYPE_I_t n_dims, 
                        DTYPE_I_t n_elems, DTYPE_I_t n_points, 
                        DTYPE_I_t[::1] npoel,
                        DTYPE_I_t[::1] nfael, DTYPE_I_t[:, ::1] lnofa, DTYPE_I_t[:, :, ::1] lpofa, 
                        DTYPE_I_t[::1] nedel, DTYPE_I_t[:, :, ::1] lpoed,
                        DTYPE_I_t[:, ::1] connectivity, DTYPE_I_t[::1] element_types):

        if n_dims < 1:
            raise ValueError("The number of dimensions must be greater than 0.")
        if n_elems < 1:
            raise ValueError("The number of elements must be greater than 0.")
        if n_points < 1:
            raise ValueError("The number of points must be greater than 0.")

        self.n_dims   = n_dims

        self.n_elems  = n_elems
        self.n_points = n_points

        self.n_faces = 0
        self.n_edges = 0

        def _validate_shape(array, expected_shape):
            if array.shape != expected_shape:
                raise ValueError(f"The array must have shape {expected_shape}, not {array.shape}.")
            return array.copy()

        _validate_shape(npoel, (NINPOL_NUM_ELEMENT_TYPES,))
        self.npoel = npoel.copy()

        _validate_shape(nfael, (NINPOL_NUM_ELEMENT_TYPES,))
        self.nfael = nfael.copy()

        _validate_shape(lnofa, (NINPOL_NUM_ELEMENT_TYPES, NINPOL_MAX_FACES_PER_ELEMENT))
        self.lnofa = lnofa.copy()

        _validate_shape(lpofa, (NINPOL_NUM_ELEMENT_TYPES, NINPOL_MAX_FACES_PER_ELEMENT, NINPOL_MAX_POINTS_PER_FACE))
        self.lpofa = lpofa.copy()

        _validate_shape(nedel, (NINPOL_NUM_ELEMENT_TYPES,))
        self.nedel = nedel.copy()

        _validate_shape(lpoed, (NINPOL_NUM_ELEMENT_TYPES, NINPOL_MAX_EDGES_PER_ELEMENT, NINPOL_MAX_POINTS_PER_EDGE))
        self.lpoed = lpoed.copy()

        self.inpoel         = connectivity.copy()
        self.element_types  = element_types.copy()
        
        self.are_elements_loaded = True
        self.are_coords_loaded = False
        
        # Set every other member memory_slice to zero
        self.esup               = np.zeros(0,       dtype=DTYPE_I)
        self.esup_ptr           = np.zeros(0,       dtype=DTYPE_I)

        self.psup               = np.zeros(0,       dtype=DTYPE_I)
        self.psup_ptr           = np.zeros(0,       dtype=DTYPE_I)

        self.inpofa             = np.zeros((0, 0),  dtype=DTYPE_I)
        self.infael             = np.zeros((0, 0),  dtype=DTYPE_I)

        self.esufa              = np.zeros(0,       dtype=DTYPE_I)
        self.esufa_ptr          = np.zeros(0,       dtype=DTYPE_I)

        self.esuel              = np.zeros((0, 0),  dtype=DTYPE_I)
        
        self.inpoed             = np.zeros((0, 0),  dtype=DTYPE_I)
        self.inedel             = np.zeros((0, 0),  dtype=DTYPE_I)

        self.point_coords       = np.zeros((0, 0),  dtype=DTYPE_F)
        self.centroids          = np.zeros((0, 0),  dtype=DTYPE_F)



    cpdef void build(self):
        
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
            int elem_type

        # Reshape the arrays
        self.esup_ptr = np.zeros(self.n_points+1, dtype=DTYPE_I)

        # Count the number of elements surrounding each point    
        for i in range(self.n_elems):
            elem_type = self.element_types[i]
            for j in range(self.npoel[elem_type]):
                self.esup_ptr[self.inpoel[i, j] + 1] += 1
        
        # Compute the cumulative sum of the number of elements surrounding each point
        for i in range(self.n_points):
            self.esup_ptr[i + 1] += self.esup_ptr[i]

        # Fill the esup array
        self.esup = np.zeros(self.esup_ptr[self.n_points], dtype=DTYPE_I)
        for i in range(self.n_elems):
            elem_type = self.element_types[i]
            for j in range(self.npoel[elem_type]):
                self.esup[self.esup_ptr[self.inpoel[i, j]]] = i
                self.esup_ptr[self.inpoel[i, j]] += 1
        
        for i in range(self.n_points, 0, -1):
            self.esup_ptr[i] = self.esup_ptr[i-1]
        self.esup_ptr[0] = 0

    cdef void build_psup(self):

        cdef:
            int i, j, k
            int stor_ptr = 0, point_idx
            int elem_type_esup

            DTYPE_I_t[::1] temp_psup = np.ones(self.n_points, dtype=DTYPE_I) * -1
        
        self.psup_ptr = np.zeros(self.n_points+1, dtype=DTYPE_I)
        self.psup_ptr[0] = 0
    
        # Upper bound for the number of points surrounding each point, pehaps this can be improved
        self.psup = np.zeros((self.esup_ptr[self.n_points] * (7)), dtype=DTYPE_I) 

        # Calculate the points surrounding each point, using temp_psup to avoid duplicates
        for i in range(self.n_points):
            for j in range(self.esup_ptr[i], self.esup_ptr[i+1]):

                elem_type_esup = self.element_types[self.esup[j]]

                for k in range(self.npoel[elem_type_esup]):
                    point_idx = self.inpoel[self.esup[j], k]
                    if point_idx != i and temp_psup[point_idx] != i:
                        self.psup[stor_ptr] = point_idx
                        temp_psup[point_idx] = i
                        stor_ptr += 1
                        
            self.psup_ptr[i+1] = stor_ptr

        # Resize the psup array to remove padding
        self.psup = self.psup[:stor_ptr]
    
    cdef void build_infael(self):
        cdef:
            int i, j, k
            int elem_type

            # Stores the points (in relation to the global point index) of the face
            DTYPE_I_t[::1] elem_face = np.zeros(NINPOL_MAX_POINTS_PER_FACE, dtype=DTYPE_I)
            DTYPE_I_t[::1] sorted_elem_face = np.zeros(NINPOL_MAX_POINTS_PER_FACE, dtype=DTYPE_I)

            # Stores the index of the points in the element (in relation to the element's local point index)
            DTYPE_I_t[::1] elem_face_index = np.zeros(NINPOL_MAX_POINTS_PER_FACE, dtype=DTYPE_I)

            # Stores the string representation of the face, for hashing
            str elem_face_str
            dict faces_dict = {}

            int current_face_index = 0
            int unused_spaces
            int face_index = 0

            int faces_upper_bound = self.n_elems * NINPOL_MAX_FACES_PER_ELEMENT

        self.inpofa = np.ones((faces_upper_bound, NINPOL_MAX_POINTS_PER_FACE), dtype=DTYPE_I) * -1
        self.infael = np.ones((self.n_elems, NINPOL_MAX_FACES_PER_ELEMENT), dtype=DTYPE_I) * -1

        # For each element
        for i in range(self.n_elems):
            elem_type = self.element_types[i]

            # For each face
            for j in range(self.nfael[elem_type]):
                elem_face_str = ""
                elem_face_index = self.lpofa[elem_type, j]

                for k in range(self.lnofa[elem_type, j]):
                    elem_face[k] = self.inpoel[i, elem_face_index[k]]

                unused_spaces = NINPOL_MAX_POINTS_PER_FACE - self.lnofa[elem_type, j]

                for k in range(unused_spaces):
                    elem_face[NINPOL_MAX_POINTS_PER_FACE - k - 1] = -1

                sorted_elem_face = np.sort(elem_face)
                for k in range(self.lnofa[elem_type, j]):
                    elem_face_str += str(sorted_elem_face[k + unused_spaces]) + ","
                    

                if faces_dict.get(elem_face_str) is None:
                    face_index = current_face_index
                    current_face_index += 1
                    faces_dict[elem_face_str] = face_index

                    for k in range(self.lnofa[elem_type, j]):
                        self.inpofa[face_index, k] = elem_face[k]

                else:
                    face_index = faces_dict[elem_face_str]
                self.infael[i, j] = face_index
        
        self.n_faces = current_face_index
        self.inpofa = self.inpofa[:self.n_faces]
    
    cdef void build_esufa(self):
        cdef:
            int i, j
            int elem_type


        self.esufa = np.ones((self.n_faces * NINPOL_MAX_ELEMENTS_PER_FACE), dtype=DTYPE_I) * -1
        self.esufa_ptr = np.zeros(self.n_faces+1, dtype=DTYPE_I)

        # Compute cumulative sum of the number of elements surrounding each face
        for i in range(self.n_elems):
            elem_type = self.element_types[i]
            for j in range(self.nfael[elem_type]):
                face = self.infael[i, j]
                self.esufa_ptr[self.infael[i, j] + 1] += 1


    cdef void build_esuel(self):
        
        # Declare every variable
        cdef:
            int j, k, l, m
            int ielem, jelem
            int ielem_type, jelem_type
            int unused_spaces
            DTYPE_I_t[::1] ielem_face = np.zeros(NINPOL_MAX_POINTS_PER_FACE, dtype=DTYPE_I)
            DTYPE_I_t[::1] jelem_face = np.zeros(NINPOL_MAX_POINTS_PER_FACE, dtype=DTYPE_I)

            DTYPE_I_t[::1] ielem_face_index = np.zeros(NINPOL_MAX_POINTS_PER_FACE, dtype=DTYPE_I)
            DTYPE_I_t[::1] jelem_face_index = np.zeros(NINPOL_MAX_POINTS_PER_FACE, dtype=DTYPE_I)

            int point, kpoint
            int num_elems, num_elems_min
            int found_elem
            int jelem_face_point
            int is_equal 

            
        self.esuel = np.ones((self.n_elems, NINPOL_MAX_FACES_PER_ELEMENT), dtype=DTYPE_I) * -1

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
                
                unused_spaces = NINPOL_MAX_POINTS_PER_FACE - self.lnofa[ielem_type, j]

                for k in range(unused_spaces):
                    ielem_face[NINPOL_MAX_POINTS_PER_FACE - k - 1] = -1

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

                                for m in range(self.lnofa[ielem_type, j]):
                                    if jelem_face_point == ielem_face[m]:
                                        is_equal += 1
                                        break
                        
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
        
        cdef:
            int i, j, k
            int elem_type

            # Stores the points (in relation to the global point index) of the edge
            DTYPE_I_t[::1] elem_edge = np.zeros(NINPOL_MAX_POINTS_PER_EDGE, dtype=DTYPE_I)
            DTYPE_I_t[::1] sorted_elem_edge = np.zeros(NINPOL_MAX_POINTS_PER_EDGE, dtype=DTYPE_I)

            # Stores the index of the points in the element (in relation to the element's local point index)
            DTYPE_I_t[::1] elem_edge_index = np.zeros(NINPOL_MAX_POINTS_PER_EDGE, dtype=DTYPE_I)

            # Stores the string representation of the edge, for hashing
            str elem_edge_str
            dict edges_dict = {}

            int current_edge_index = 0
            int edge_index = 0

        self.inedel = np.ones((self.n_elems, NINPOL_MAX_EDGES_PER_ELEMENT), dtype=DTYPE_I) * -1
        self.inpoed = np.ones((self.n_elems * NINPOL_MAX_EDGES_PER_ELEMENT, 
                               NINPOL_MAX_POINTS_PER_EDGE), dtype=DTYPE_I) * -1
        
        # For each element
        for i in range(self.n_elems):
            elem_type = self.element_types[i]

            # For each edge
            for j in range(self.nedel[elem_type]):
                elem_edge_str = ""
                elem_edge_index = self.lpoed[elem_type, j]

                # Assume there's the exacly same amount of points in each edge (usually 2)
                for k in range(NINPOL_MAX_POINTS_PER_EDGE):
                    elem_edge[k] = self.inpoel[i, elem_edge_index[k]]

                sorted_elem_edge = np.sort(elem_edge)
                for k in range(NINPOL_MAX_POINTS_PER_EDGE):
                    elem_edge_str += str(sorted_elem_edge[k]) + ","
                
                if edges_dict.get(elem_edge_str) is None:
                    edge_index = current_edge_index
                    current_edge_index += 1
                    edges_dict[elem_edge_str] = edge_index

                    for k in range(NINPOL_MAX_POINTS_PER_EDGE):
                        self.inpoed[edge_index, k] = elem_edge[k]

                else:
                    edge_index = edges_dict[elem_edge_str]
                self.inedel[i, j] = edge_index
        
        self.n_edges = current_edge_index
        self.inpoed = self.inpoed[:self.n_edges]

                    

    
    cdef void load_point_coords(self, const DTYPE_F_t[:, ::1] coords):
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
            int elem_type
            int npoel_e
        self.centroids = np.zeros((self.n_elems, self.n_dims), dtype=DTYPE_F)
        
        cdef:
            int use_threads = self.n_elems > 1000
            float machine_epsilon = 10 ** int(np.log10(np.finfo(DTYPE_F).eps))

        omp_set_num_threads(8 if use_threads else 1)

        for i in prange(self.n_elems, nogil=True, schedule='static', num_threads=8 if use_threads else 1):
            elem_type = self.element_types[i]
            npoel_e = self.npoel[elem_type]
            for j in range(npoel_e):
                for k in range(self.n_dims):
                    self.centroids[i, k] += self.point_coords[self.inpoel[i, j], k] / npoel_e

        