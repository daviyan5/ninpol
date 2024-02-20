import numpy as np
import numba as nb
import yaml


class Grid:
    def __init__(self, n_dims, n_elems, n_points):
        self.n_dims         = n_dims
        self.n_elems        = n_elems
        self.n_points       = n_points
        point_ordering_obj  = yaml.safe_load(open("./ninpol/utils/point_ordering.yaml"))
        point_ordering = point_ordering_obj['elements']

        nfael = np.zeros(16, dtype=np.int32)
        lnofa = np.zeros((16, 6), dtype=np.int32)
        lpofa = np.ones((16, 6, 4), dtype=np.int32) * -1

        self.n_point_to_type = np.zeros(16, dtype=np.int32)

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

    def build(self, connectivity, elem_types = None):
        return_values = build(self.n_dims, self.n_points, self.n_elems, self.n_point_to_type, self.nfael, self.lnofa, self.lpofa, connectivity, elem_types)
        self.inpoel, self.n_points_per_elem, self.element_types, self.esup, self.esup_ptr, self.psup, self.psup_ptr, self.esuel = return_values

def build(n_dims, n_points, n_elems, n_point_to_type, nfael, lnofa, lpofa, connectivity, elem_types = None):
    # Check that the connectivity matrix is not None and has the correct shape
    if connectivity is None:
        raise ValueError("The connectivity matrix cannot be None.")
    if connectivity.shape[0] != n_elems:
        raise ValueError("The number of rows in the connectivity matrix must be equal to the number of elements.")
    
    inpoel = connectivity
    
    n_points_per_elem = np.array([len(np.argwhere(inpoel[i] != -1)) for i in range(n_elems)], dtype=np.int32)
    element_types = elem_types if elem_types is not None else n_point_to_type[n_points_per_elem]

    esup, esup_ptr = build_esup(n_elems, n_points, inpoel, n_points_per_elem)
    psup, psup_ptr = build_psup(n_points, inpoel, esup, esup_ptr)
    esuel = build_esuel(n_points, n_elems, inpoel, element_types, nfael, lnofa, lpofa, esup, esup_ptr)
    if n_dims == 3:
        build_inpoed()
        build_ledel()

    return inpoel, n_points_per_elem, element_types, esup, esup_ptr, psup, psup_ptr, esuel

@nb.njit()
def build_esup(n_elems, n_points, inpoel, n_points_per_elem):
    esup_ptr = np.zeros(n_points + 1, dtype=np.int32)
    for i in range(n_elems):
        for j in range(n_points_per_elem[i]):
            esup_ptr[inpoel[i, j] + 1] += 1
    esup_ptr = np.cumsum(esup_ptr)  
    esup = np.zeros(esup_ptr[n_points], dtype=np.int32)
    for i in range(n_elems):
        for j in range(n_points_per_elem[i]):
            esup[esup_ptr[inpoel[i, j]]] = i
            esup_ptr[inpoel[i, j]] += 1
    for i in range(n_points, 0, -1):
        esup_ptr[i] = esup_ptr[i-1]

    esup_ptr[0] = 0
    return esup, esup_ptr

@nb.njit()
def build_psup(n_points, inpoel, esup, esup_ptr):
    psup_ptr = np.zeros(n_points + 1, dtype=np.int32)
    psup = np.zeros(esup_ptr[n_points] * (7), dtype=np.int32)
    stor_ptr = 0

    for i in range(n_points):
        elems = esup[esup_ptr[i]:esup_ptr[i+1]]
        if elems.shape[0] == 0:
            continue
        x = inpoel[elems, :].flatten()
        mx    = np.max(x) + 1
        used  = np.zeros(mx, dtype=np.uint8)
        used[x] = 1
        points = np.argwhere(used == 1)[:,0]
        points = points[points != i]
        psup[stor_ptr:stor_ptr + points.shape[0]] = points
        psup_ptr[i+1] = psup_ptr[i] + points.shape[0]
        stor_ptr += points.shape[0]

    psup = psup[:psup_ptr[-1]]

    return psup, psup_ptr

@nb.njit()
def build_esuel(n_points, n_elems, inpoel, element_types, nfael, lnofa, lpofa, esup, esup_ptr):
    esuel = np.ones((n_elems, 6), dtype=np.int32) * -1
    # For each element
    ielem_face = np.zeros(4, dtype=np.int32)
    ielem_face_index = np.zeros(4, dtype=np.int32)
    jelem_face_index = np.zeros(4, dtype=np.int32)

    for ielem in range(n_elems):
        ielem_type = element_types[ielem]

        # For each face
        for j in range(nfael[ielem_type]):

            if esuel[ielem, j] != -1:
                continue
            # Choose a point from the face
            ielem_face_index = lpofa[ielem_type, j]
            for k in range(lnofa[ielem_type, j]):
                ielem_face[k] = inpoel[ielem, ielem_face_index[k]]

            for k in range (4 - lnofa[ielem_type, j]):
                ielem_face[3 - k] = -1
            point = ielem_face[0]
            num_elems_min = esup_ptr[point+1] - esup_ptr[point]

            # Choose the point with the least number of elements around it
            for k in range(lnofa[ielem_type, j]):
                kpoint = ielem_face[k]
                num_elems = esup_ptr[kpoint+1] - esup_ptr[kpoint]
                if num_elems < num_elems_min:
                    point = kpoint
                    num_elems_min = num_elems


            found_elem = False

            # For each element around the point
            for k in range(esup_ptr[point], esup_ptr[point+1]):
                jelem = esup[k]
                jelem_type = element_types[jelem]

                # If the element around the point is not the current element
                if jelem != ielem:

                    # For each face of the element around the point
                    for l in range(nfael[jelem_type]):
                        is_equal = 0
                        jelem_face_index = lpofa[jelem_type, l]

                        for m in range(lnofa[jelem_type, l]):
                            jelem_face_point = inpoel[jelem, jelem_face_index[m]]
                            if (jelem_face_point == ielem_face[0] or 
                                jelem_face_point == ielem_face[1] or 
                                jelem_face_point == ielem_face[2] or 
                                jelem_face_point == ielem_face[3]):
                                is_equal += 1

                            
                        # If the face of the element around the point is equal to the face of the current element
                        if is_equal == lnofa[ielem_type, j]:

                            # Add the element around the point to the list of elements around the current element
                            esuel[ielem, j] = jelem
                            # Add the current element to the list of elements around the element around the point
                            esuel[jelem, l] = ielem

                            found_elem = True
                        
                        
                        if found_elem:
                            break


                if found_elem:
                    break
    
    return esuel

@nb.njit()
def build_inpoed():
    pass

@nb.njit()
def build_ledel():
    pass
