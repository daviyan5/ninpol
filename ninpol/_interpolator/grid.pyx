"""
This file contains the "Grid" class implementation
"""

cimport openmp

import numpy as np

from cython.parallel import prange
from libc.math cimport sqrt
from libc.stdio cimport printf
from cython cimport typeof

DTYPE_I = np.int64
DTYPE_F = np.float64


from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport sort
from libcpp.string cimport string, to_string
from ctypes import sizeof as csizeof

from ..utils.robin_hood cimport unordered_map

cdef size_t myhash(DTYPE_I_t[::1] vec, int unused_spaces) noexcept nogil:
  cdef:
    int i
    int x
    size_t seed = len(vec)
    

  for i in range(unused_spaces, len(vec)):
    x = vec[i]
    x = ((x >> 16) ^ x) * <unsigned int> 0x45d9f3b
    x = ((x >> 16) ^ x) * <unsigned int> 0x45d9f3b
    x = (x >> 16) ^ x
    seed ^= x + <unsigned int> 0x9e3779b9 + (seed << 6) + (seed >> 2)
  
  return seed


cdef class Grid:
    def __cinit__(self, DTYPE_I_t dim, 
                        DTYPE_I_t n_elems, DTYPE_I_t n_points, 
                        DTYPE_I_t[::1] npoel,
                        DTYPE_I_t[::1] nfael, DTYPE_I_t[:, ::1] lnofa, DTYPE_I_t[:, :, ::1] lpofa, 
                        DTYPE_I_t[::1] nedel, DTYPE_I_t[:, :, ::1] lpoed,
                        DTYPE_I_t[:, ::1] connectivity, DTYPE_I_t[::1] element_types,
                        int logging = False, int build_edges = False):

        if dim < 1:
            raise ValueError("The number of dimensions must be greater than 0.")
        if n_elems < 1:
            raise ValueError("The number of elements must be greater than 0.")
        if n_points < 1:
            raise ValueError("The number of points must be greater than 0.")

        self.dim   = dim

        self.n_elems  = n_elems
        self.n_points = n_points

        self.n_faces = 0
        self.n_edges = 0

        self.MX_ELEMENTS_PER_POINT = 0
        self.MX_POINTS_PER_POINT = 0
        self.MX_ELEMENTS_PER_FACE = 0
        self.MX_FACES_PER_POINT = 0
        
        self.logging = logging
        self.logger  = Logger("Grid")

        self.build_edges = build_edges

        def _validate_shape(array, expected_shape):
            if array.shape != expected_shape:
                raise ValueError(f"The array must have shape {expected_shape}, not {array.shape}.")
            return array.copy()

        _validate_shape(npoel, (NinpolSizes.NINPOL_NUM_ELEMENT_TYPES,))
        self.npoel = npoel.copy()

        _validate_shape(nfael, (NinpolSizes.NINPOL_NUM_ELEMENT_TYPES,))
        self.nfael = nfael.copy()

        _validate_shape(lnofa, (NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, NinpolSizes.NINPOL_MAX_FACES_PER_ELEMENT))
        self.lnofa = lnofa.copy()

        _validate_shape(lpofa, (NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, NinpolSizes.NINPOL_MAX_FACES_PER_ELEMENT, NinpolSizes.NINPOL_MAX_POINTS_PER_FACE))
        self.lpofa = lpofa.copy()

        _validate_shape(nedel, (NinpolSizes.NINPOL_NUM_ELEMENT_TYPES,))
        self.nedel = nedel.copy()

        _validate_shape(lpoed, (NinpolSizes.NINPOL_NUM_ELEMENT_TYPES, NinpolSizes.NINPOL_MAX_EDGES_PER_ELEMENT, NinpolSizes.NINPOL_MAX_POINTS_PER_EDGE))
        self.lpoed = lpoed.copy()

        self.inpoel         = connectivity.copy()
        self.element_types  = element_types.copy()
        
        self.are_elements_loaded      = True
        self.are_coords_loaded        = False
        self.are_structures_built     = False
        self.are_centroids_calculated = False
        self.are_normals_calculated   = False

        self.boundary_faces  = np.zeros(0, dtype=DTYPE_I)
        self.boundary_points = np.zeros(0, dtype=DTYPE_I)
        # Set every other member memory_slice to zero
        self.esup         = np.zeros(0,       dtype=DTYPE_I)
        self.esup_ptr     = np.zeros(0,       dtype=DTYPE_I)

        self.psup         = np.zeros(0,       dtype=DTYPE_I)
        self.psup_ptr     = np.zeros(0,       dtype=DTYPE_I)

        self.inpofa       = np.zeros((0, 0),  dtype=DTYPE_I)
        self.infael       = np.zeros((0, 0),  dtype=DTYPE_I)

        self.esuf         = np.zeros(0,       dtype=DTYPE_I)
        self.esuf_ptr     = np.zeros(0,       dtype=DTYPE_I)

        self.fsup         = np.zeros(0,       dtype=DTYPE_I)
        self.fsup_ptr     = np.zeros(0,       dtype=DTYPE_I)

        self.esuel        = np.zeros((0, 0),  dtype=DTYPE_I)
        
        self.inpoed       = np.zeros((0, 0),  dtype=DTYPE_I)
        self.inedel       = np.zeros((0, 0),  dtype=DTYPE_I)

        self.point_coords = np.zeros((0, 0),  dtype=DTYPE_F)
        self.centroids    = np.zeros((0, 0),  dtype=DTYPE_F)
        self.faces_centers = np.zeros((0, 0), dtype=DTYPE_F)
        self.faces_areas = np.zeros(0   , dtype=DTYPE_F)

        self.normal_faces = np.zeros((0, 0),  dtype=DTYPE_F)

    
    cpdef void build(self):
        
        
        # Calculate the elements surrounding each point
        cdef:
            double start_time = 0.0, end_time = 0.0
            timespec ts
        
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + ts.tv_nsec * 1e-9

        # Builds esup_ptr and esup 
        self.build_esup()
        call_name = "build esup"
        
        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + ts.tv_nsec * 1e-9
        if self.logging:
            self.logger.log(f"Time to {call_name:<15}: {end_time - start_time:.3f} s", "INFO")
        
        
        # Calculate the points surrounding each point
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + ts.tv_nsec * 1e-9

        # Builds psup and psup_ptr
        self.build_psup()
        call_name = "build_psup"

        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + ts.tv_nsec * 1e-9
        if self.logging:
            self.logger.log(f"Time to {call_name:<15}: {end_time - start_time:.3f} s", "INFO")

        # Calculate the elements surrounding each element
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + ts.tv_nsec * 1e-9
        self.build_esuel()
        call_name = "build_esuel"
        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + ts.tv_nsec * 1e-9
        if self.logging:
            self.logger.log(f"Time to {call_name:<15}: {end_time - start_time:.3f} s", "INFO")

        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + ts.tv_nsec * 1e-9

        # Calculate the faces composing each element, builds infael and inpofa
        self.build_infael()
        call_name = "build_infael"

        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + ts.tv_nsec * 1e-9
        if self.logging:
            self.logger.log(f"Time to {call_name:<15}: {end_time - start_time:.3f} s", "INFO")
        
        # Calculate the faces surrounding each point
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + ts.tv_nsec * 1e-9

        # Builds fsup and fsup_ptr
        self.build_fsup()
        call_name = "build_fsup"

        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + ts.tv_nsec * 1e-9
        if self.logging:
            self.logger.log(f"Time to {call_name:<15}: {end_time - start_time:.3f} s", "INFO")

        # Calculate the elements surrounding each face
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + ts.tv_nsec * 1e-9

        # Builds esuf and esuf_ptr
        self.build_esuf()
        call_name = "build esuf"

        clock_gettime(CLOCK_REALTIME, &ts)
        end_time = ts.tv_sec + ts.tv_nsec * 1e-9
        if self.logging:
            self.logger.log(f"Time to {call_name:<15}: {end_time - start_time:.3f} s", "INFO")

        # Calculate the edges surrounding each element
        if self.build_edges:
            if self.logging:
                self.logger.log("Grid will build edge data.", "INFO")
            clock_gettime(CLOCK_REALTIME, &ts)
            start_time = ts.tv_sec + ts.tv_nsec * 1e-9
            self.build_inedel()
            call_name = "build_inedel"
            clock_gettime(CLOCK_REALTIME, &ts)
            end_time = ts.tv_sec + ts.tv_nsec * 1e-9
            if self.logging:
                self.logger.log(f"Time to {call_name:<15}: {end_time - start_time:.3f} s", "INFO")
        elif self.logging:
                self.logger.log("Grid will not build edge data.", "INFO")

        self.are_structures_built = True
    
    cdef void build_esup(self):
        
        cdef:
            int i, j
            int elem_type
            int point

        # Reshape the arrays
        self.esup_ptr = np.zeros(self.n_points+1, dtype=DTYPE_I)

        # Count the number of elements surrounding each point    
        for i in range(self.n_elems):
            elem_type = self.element_types[i]
            for j in range(self.npoel[elem_type]):
                point = self.inpoel[i, j]
                self.esup_ptr[point + 1] += 1
                self.MX_ELEMENTS_PER_POINT = max(self.MX_ELEMENTS_PER_POINT, 
                                                  self.esup_ptr[point + 1])
        
        # Compute the cumulative sum of the number of elements surrounding each point
        for i in range(self.n_points):
            self.esup_ptr[i + 1] += self.esup_ptr[i]

        # Fill the esup array
        self.esup = np.zeros(self.esup_ptr[self.n_points], dtype=DTYPE_I)
        for i in range(self.n_elems):
            elem_type = self.element_types[i]
            for j in range(self.npoel[elem_type]):
                point = self.inpoel[i, j]
                self.esup[self.esup_ptr[point]] = i
                self.esup_ptr[point] += 1
        
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
        self.psup = np.zeros((self.esup_ptr[self.n_points] * (NinpolSizes.NINPOL_MAX_POINTS_PER_ELEMENT - 1)), dtype=DTYPE_I) 

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
            self.MX_POINTS_PER_POINT = max(self.MX_POINTS_PER_POINT,
                                            self.psup_ptr[i + 1] - self.psup_ptr[i])

        # Resize the psup array to remove padding
        self.psup = self.psup[:stor_ptr]

    cdef void build_infael(self):
        cdef:
            int i, j, k, l
            int ielem_type, kelem_type
            int face_index = 0
            int faces_upper_bound = self.n_elems * NinpolSizes.NINPOL_MAX_FACES_PER_ELEMENT
            DTYPE_I_t[:, ::1] face_to_elem = np.ones((faces_upper_bound, 2), dtype=DTYPE_I) * -1
        
        face_index = 0
        self.infael = np.ones((self.n_elems, NinpolSizes.NINPOL_MAX_FACES_PER_ELEMENT), dtype=DTYPE_I) * -1

        for i in range(self.n_elems):
            ielem_type = self.element_types[i]
            for j in range(self.nfael[ielem_type]):
                if self.infael[i, j] != -1:
                    continue
                
                self.infael[i, j] = face_index
                face_index = face_index + 1

                face_to_elem[self.infael[i, j], 0] = i
                face_to_elem[self.infael[i, j], 1] = j
                
                k = self.esuel[i, j]
                if k == -1:
                    continue
                kelem_type = self.element_types[k]
                for l in range(self.nfael[kelem_type]):
                    if self.esuel[k, l] == i:
                        self.infael[k, l] = self.infael[i, j]
                        break

        self.n_faces = face_index
        self.inpofa = np.ones((self.n_faces, NinpolSizes.NINPOL_MAX_POINTS_PER_FACE), dtype=DTYPE_I) * -1
        cdef:
            int face
        for face in range(self.n_faces):
            i = face_to_elem[face, 0]
            j = face_to_elem[face, 1]
            ielem_type = self.element_types[i]
            for k in range(self.lnofa[ielem_type, j]):
                self.inpofa[face, k] = self.inpoel[i, self.lpofa[ielem_type, j, k]]
    
    cdef void build_fsup(self):
            
        cdef:
            int i, j, k
            int elem_type
            int point

        self.fsup_ptr = np.zeros(self.n_points+1, dtype=DTYPE_I)

        for i in range(self.n_faces):
            for j in range(int(NinpolSizes.NINPOL_MAX_POINTS_PER_FACE)):
                if self.inpofa[i, j] == -1:
                    break
                point = self.inpofa[i, j]
                self.fsup_ptr[point + 1] += 1
                self.MX_FACES_PER_POINT = max(self.MX_FACES_PER_POINT, 
                                              self.fsup_ptr[point + 1])

        for i in range(self.n_points):
            self.fsup_ptr[i + 1] += self.fsup_ptr[i]

        self.fsup = np.zeros(self.fsup_ptr[self.n_points], dtype=DTYPE_I)
        for i in range(self.n_faces):
            for j in range(int(NinpolSizes.NINPOL_MAX_POINTS_PER_FACE)):
                if self.inpofa[i, j] == -1:
                    break
                point = self.inpofa[i, j]
                self.fsup[self.fsup_ptr[point]] = i
                self.fsup_ptr[point] += 1
        
        for i in range(self.n_points, 0, -1):
            self.fsup_ptr[i] = self.fsup_ptr[i-1]
        self.fsup_ptr[0] = 0

    cdef void build_esuf(self):

        # Same logic as esup
        cdef:
            int i, j, k
            int elem_type
            int face


        self.esuf_ptr = np.zeros(self.n_faces + 1, dtype=DTYPE_I)

        # Count the number of elements surrounding each face 
        for i in range(self.n_elems):
            elem_type = self.element_types[i]
            for j in range(self.nfael[elem_type]):
                face = self.infael[i, j]
                self.esuf_ptr[face + 1] += 1
                self.MX_ELEMENTS_PER_FACE = max(self.MX_ELEMENTS_PER_FACE, 
                                                self.esuf_ptr[face + 1])

        # Compute the cumulative sum of the number of elements surrounding each face
        for i in range(self.n_faces):
            self.esuf_ptr[i + 1] += self.esuf_ptr[i]
        
        # Fill the esuf array
        self.esuf = np.zeros((self.esuf_ptr[self.n_faces]), dtype=DTYPE_I)
        for i in range(self.n_elems):
            elem_type = self.element_types[i]
            for j in range(self.nfael[elem_type]):
                face = self.infael[i, j]
                self.esuf[self.esuf_ptr[face]] = i
                self.esuf_ptr[face] += 1

        for i in range(self.n_faces, 0, -1):
            self.esuf_ptr[i] = self.esuf_ptr[i-1]
        self.esuf_ptr[0] = 0
        
        cdef:
            int elem
            int use_threads = min(16, np.ceil(self.n_faces / 400))

        # Make sure that the first element is the one that defined self.inpofa
        omp_set_num_threads(use_threads)
        for face in prange(self.n_faces, nogil=True, schedule='static', num_threads=use_threads):
            elem = self.esuf[self.esuf_ptr[face]]
            if elem != -1:
                elem_type = self.element_types[elem]
                for j in range(self.nfael[elem_type]):
                    if self.infael[elem, j] == face:
                        break 
                for k in range(self.lnofa[elem_type, j]):
                    self.inpofa[face, k] = self.inpoel[elem, self.lpofa[elem_type, j, k]]
        
        self.boundary_faces = np.zeros(self.n_faces, dtype=DTYPE_I)
        self.boundary_points = np.zeros(self.n_points, dtype=DTYPE_I)
        
        omp_set_num_threads(use_threads)
        for i in prange(self.n_faces, nogil=True, schedule='static', num_threads=use_threads):
            if self.esuf_ptr[i + 1] - self.esuf_ptr[i] == 1:
                self.boundary_faces[i] = True
                for j in self.inpofa[i]:
                    if j == -1:
                        break
                    self.boundary_points[j] = True

        


    cdef void build_esuel(self):
        
        # Declare every variable
        cdef:
            int j, k, l, m, o
            int ielem, jelem
            int ielem_type, jelem_type
            int unused_spaces

            int point, kpoint
            int num_elems, num_elems_min
            int found_elem
            int jelem_face_point
            int is_equal 

            int use_threads = min(16, np.ceil(self.n_elems / 400))

            
        self.esuel = np.ones((self.n_elems, NinpolSizes.NINPOL_MAX_FACES_PER_ELEMENT), dtype=DTYPE_I) * -1

        # For each element
        omp_set_num_threads(use_threads)
        for ielem in prange(self.n_elems, nogil=True, schedule='static', num_threads=use_threads):
            ielem_type = self.element_types[ielem]
            # For each face
            for j in range(self.nfael[ielem_type]):

                if self.esuel[ielem, j] != -1:
                    continue
                
                point = self.inpoel[ielem, self.lpofa[ielem_type, j, 0]]
                num_elems_min = self.esup_ptr[point+1] - self.esup_ptr[point]

                # Choose the point with the least number of elements around it
                for k in range(self.lnofa[ielem_type, j]):
                    kpoint = self.inpoel[ielem, self.lpofa[ielem_type, j, k]]
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
                            is_equal = 0
                            for m in range(self.lnofa[jelem_type, l]):
                                jelem_face_point = self.inpoel[jelem, self.lpofa[jelem_type, l, m]]

                                for o in range(self.lnofa[ielem_type, j]):
                                    if jelem_face_point == self.inpoel[ielem, self.lpofa[ielem_type, j, o]]:
                                        is_equal = is_equal + 1
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
            size_t hash_key

            # Stores the points (in relation to the global point index) of the edge
            DTYPE_I_t[::1] elem_edge = np.zeros(NinpolSizes.NINPOL_MAX_POINTS_PER_EDGE, dtype=DTYPE_I)
            DTYPE_I_t[::1] sorted_elem_edge = np.zeros(NinpolSizes.NINPOL_MAX_POINTS_PER_EDGE, dtype=DTYPE_I)

            # Stores the string representation of the edge, for hashing
            unordered_map[int, int] edges_dict

            int current_edge_index = 0
            int edge_index = 0

            int MAX_EDGES_PER_ELEMENT = NinpolSizes.NINPOL_MAX_EDGES_PER_ELEMENT
            int MAX_POINTS_PER_EDGE = NinpolSizes.NINPOL_MAX_POINTS_PER_EDGE

        self.inedel = np.ones((self.n_elems, MAX_EDGES_PER_ELEMENT), dtype=DTYPE_I) * -1
        self.inpoed = np.ones((self.n_elems * MAX_EDGES_PER_ELEMENT, 
                               MAX_POINTS_PER_EDGE), dtype=DTYPE_I) * -1
        
        # For each element
        for i in range(self.n_elems):
            elem_type = self.element_types[i]

            # For each edge
            for j in range(self.nedel[elem_type]):
                
                # Assume there's the exacly same amount of points in each edge (usually 2)
                for k in range(int(NinpolSizes.NINPOL_MAX_POINTS_PER_EDGE)):
                    elem_edge[k] = self.inpoel[i, self.lpoed[elem_type, j, k]]
                    sorted_elem_edge[k] = elem_edge[k]

                if elem_edge[0] > elem_edge[1]:
                    sorted_elem_edge[0], sorted_elem_edge[1] = sorted_elem_edge[1], sorted_elem_edge[0]
                
                hash_key = myhash(sorted_elem_edge, 0)
                
                if edges_dict.count(hash_key) == 0:
                    edge_index = edges_dict.size()
                    edges_dict[hash_key] = edge_index

                    for k in range(int(NinpolSizes.NINPOL_MAX_POINTS_PER_EDGE)):
                        self.inpoed[edge_index, k] = elem_edge[k]

                else:
                    edge_index = edges_dict[hash_key]
                self.inedel[i, j] = edge_index
        
        self.n_edges = edges_dict.size()
        self.inpoed = self.inpoed[:self.n_edges]

                    
    def get_data(self):
        """
        Returns the data of the grid as np.arrays.
        If the mesh is big, this method can be slow, as it does not cache the data and extensive copying is done.
        """
        import warnings

        cdef:
            dict data

        if self.are_coords_loaded == False:
            warnings.warn("The point coordinates have not been set.")
        if self.are_structures_built == False:
            raise ValueError("The structures have not been built.")
        if self.are_centroids_calculated == False:
            warnings.warn("The centroids have not been calculated.")

        data = {
            'n_elems':                  self.n_elems,
            'n_points':                 self.n_points,
            'n_faces':                  self.n_faces,
            'n_edges':                  self.n_edges,
            'MX_ELEMENTS_PER_POINT':    self.MX_ELEMENTS_PER_POINT,
            'MX_POINTS_PER_POINT':      self.MX_POINTS_PER_POINT,
            'MX_ELEMENTS_PER_FACE':     self.MX_ELEMENTS_PER_FACE,
            'MX_FACES_PER_POINT':       self.MX_FACES_PER_POINT,
            'point_coords':             self.point_coords.copy(),
            'centroids':                self.centroids.copy(),
            'normal_faces':             self.normal_faces.copy(),
            'faces_centers':            self.faces_centers.copy(),
            'faces_areas':              self.faces_areas.copy(),
            'boundary_faces':           self.boundary_faces.copy(),
            'boundary_points':          self.boundary_points.copy(),
            'inpoel':                   self.inpoel.copy(),
            'element_types':            self.element_types.copy(),
            'inpofa':                   self.inpofa.copy(),
            'infael':                   self.infael.copy(),
            'inpoed':                   self.inpoed.copy(),
            'inedel':                   self.inedel.copy(),
            'point_coords':             self.point_coords.copy(),
            'centroids':                self.centroids.copy()
        }

        # Converts the data that needs pointers (e.g esup) to bi-dimensional arrays
        cdef:
            int i, j

            DTYPE_I_t[:, ::1] esup2d = np.ones((self.n_points, self.MX_ELEMENTS_PER_POINT), dtype=DTYPE_I) * -1
            DTYPE_I_t[:, ::1] psup2d = np.ones((self.n_points, self.MX_POINTS_PER_POINT),   dtype=DTYPE_I) * -1
            DTYPE_I_t[:, ::1] esuf2d = np.ones((self.n_faces, self.MX_ELEMENTS_PER_FACE),   dtype=DTYPE_I) * -1
            DTYPE_I_t[:, ::1] fsup2d = np.ones((self.n_points, self.MX_FACES_PER_POINT),    dtype=DTYPE_I) * -1

        for i in range(self.n_points):
            for j in range(self.esup_ptr[i], self.esup_ptr[i+1]):
                esup2d[i, j - self.esup_ptr[i]] = self.esup[j]

            for j in range(self.psup_ptr[i], self.psup_ptr[i+1]):
                psup2d[i, j - self.psup_ptr[i]] = self.psup[j]
            
            for j in range(self.fsup_ptr[i], self.fsup_ptr[i+1]):
                fsup2d[i, j - self.fsup_ptr[i]] = self.fsup[j]
        
        for i in range(self.n_faces):
            for j in range(self.esuf_ptr[i], self.esuf_ptr[i+1]):
                esuf2d[i, j - self.esuf_ptr[i]] = self.esuf[j]

        data['esup']  = esup2d.copy()
        data['psup']  = psup2d.copy()
        data['esuf']  = esuf2d.copy()
        data['fsup']  = fsup2d.copy()

        for key in data:
            if not isinstance(data[key], int) and not isinstance(data[key], float):
                data[key] = np.asarray(data[key])

        return data
        
    
    cdef void load_point_coords(self, const DTYPE_F_t[:, ::1] coords):
        if coords.shape[1] != 3:
            # Pads every point with zeros
            self.point_coords = np.zeros((coords.shape[0], 3), dtype=DTYPE_F)
            self.point_coords[:, :coords.shape[1]] = coords.copy()
        self.point_coords = coords.copy()
        self.are_coords_loaded = True

    cdef void calculate_centroids(self):
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
        # Then, using esuf, we can calculate the volume and centroid of each element

        # However, for now, we will use only the average of the points

        cdef:
            int i, j, k
            int elem_type
            int npoel_e
            int npofa

        self.centroids = np.zeros((self.n_elems, 3), dtype=DTYPE_F)
        
        cdef:
            float machine_epsilon = 10 ** int(np.log10(np.finfo(DTYPE_F).eps))


        cdef int use_threads = min(16, np.ceil(self.n_elems / 400))
        omp_set_num_threads(use_threads)
        for i in prange(self.n_elems, nogil=True, schedule='static', num_threads=use_threads):
            elem_type = self.element_types[i]
            npoel_e = self.npoel[elem_type]
            for j in range(npoel_e):
                for k in range(self.dim):
                    self.centroids[i, k] += self.point_coords[self.inpoel[i, j], k] / npoel_e

        self.faces_centers = np.zeros((self.n_faces, 3), dtype=DTYPE_F)
        omp_set_num_threads(use_threads)
        for i in prange(self.n_faces, nogil=True, schedule='static', num_threads=use_threads):
            npofa = 0
            for j in range(int(NinpolSizes.NINPOL_MAX_POINTS_PER_FACE)):
                if self.inpofa[i, j] == -1:
                    break
                npofa = npofa + 1
                for k in range(self.dim):
                    self.faces_centers[i, k] += self.point_coords[self.inpofa[i, j], k]
            for k in range(self.dim):
                self.faces_centers[i, k] /= npofa
            
        self.are_centroids_calculated = True

    cdef void calculate_normal_faces(self):
        # For each face, select the 3 first points and calculate the normal to the face.
        # Then, normalize it

        self.normal_faces = np.zeros((self.n_faces, 3), dtype=DTYPE_F)
        cdef:
            int i, j, k
            int face = 0
            int elem
            int point1 = 0, point2 = 0, point3 = 0, point4 = 0

            float v1x, v1y, v1z, v2x, v2y, v2z
            float elemx, elemy, elemz
            float normalx, normaly, normalz

            float norm = 0.0
            int use_threads = min(16, np.ceil(self.n_faces / 400))

            int npofa = 0

        self.faces_areas = np.zeros(self.n_faces, dtype=DTYPE_F)
        if self.dim == 3:
            omp_set_num_threads(use_threads)
            for face in prange(self.n_faces, nogil=True, schedule='static', num_threads=use_threads):
                npofa  = 3 if self.inpofa[face, 3] == -1 else 4
                point1 = self.inpofa[face, 0]
                point2 = self.inpofa[face, 1]
                point3 = self.inpofa[face, 2]
                
                v1x = self.point_coords[point1, 0] - self.point_coords[point2, 0]
                v1y = self.point_coords[point1, 1] - self.point_coords[point2, 1]
                v1z = self.point_coords[point1, 2] - self.point_coords[point2, 2]

                v2x = self.point_coords[point3, 0] - self.point_coords[point2, 0]
                v2y = self.point_coords[point3, 1] - self.point_coords[point2, 1]
                v2z = self.point_coords[point3, 2] - self.point_coords[point2, 2]

                normalx = v1y * v2z - v1z * v2y
                normaly = v1z * v2x - v1x * v2z
                normalz = v1x * v2y - v1y * v2x

                norm = sqrt(normalx * normalx + normaly * normaly + normalz * normalz)
                norm = abs(norm)
                
                self.normal_faces[face, 0] = normalx / norm
                self.normal_faces[face, 1] = normaly / norm
                self.normal_faces[face, 2] = normalz / norm
                if npofa == 3:
                    self.faces_areas[face] = norm / 2.0
                else:
                    # since the first triangle was 1 - 2 and 3 - 2, the second triangle is 1 - 4 and 3 - 4
                    point4 = self.inpofa[face, 3]
                    v1x = self.point_coords[point1, 0] - self.point_coords[point4, 0]
                    v1y = self.point_coords[point1, 1] - self.point_coords[point4, 1]
                    v1z = self.point_coords[point1, 2] - self.point_coords[point4, 2]

                    v2x = self.point_coords[point3, 0] - self.point_coords[point4, 0]
                    v2y = self.point_coords[point3, 1] - self.point_coords[point4, 1]
                    v2z = self.point_coords[point3, 2] - self.point_coords[point4, 2]

                    normalx = v1y * v2z - v1z * v2y
                    normaly = v1z * v2x - v1x * v2z
                    normalz = v1x * v2y - v1y * v2x

                    self.faces_areas[face] = (norm + sqrt(normalx * normalx + normaly * normaly + normalz * normalz)) / 2.0
        else:
            omp_set_num_threads(use_threads)
            for face in prange(self.n_faces, nogil=True, schedule='static', num_threads=use_threads):
                # Faces are edges in 2D
                point1 = self.inpofa[face, 0]
                point2 = self.inpofa[face, 1] 

                v1x = self.point_coords[point1, 0] - self.point_coords[point2, 0]
                v1y = self.point_coords[point1, 1] - self.point_coords[point2, 1]
                
                normalx = -v1y
                normaly = v1x

                norm = sqrt(normalx * normalx + normaly * normaly)
                norm = abs(norm)

                self.normal_faces[face, 0] = normalx / norm
                self.normal_faces[face, 1] = normaly / norm
                self.normal_faces[face, 2] = 0.0

                self.faces_areas[face] = norm
        
        
        self.are_normals_calculated = True
