"""
This file contains the "Grid" class definition, for mesh manipulation. 
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
from .ninpol_defines cimport *
cimport numpy as cnp
cnp.import_array()                  # Needed to use NumPy C API

from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t



cdef class Grid:
    """
    Stores and manipulates the mesh data.
    This is a class intended to be used only from Cython, specifically from the 'interpolator.pyx' file.

    Attributes
    ----------
    n_dims : int
        Number of dimensions
    n_elems : int
        Number of elements
    n_points : int
        Number of points (vertices) that compose the mesh
    n_faces: int
        Number of faces that compose the mesh
    n_points_per_elem : numpy.ndarray
        Number of points per element

    element_types : numpy.ndarray
        Array containing the type of each element. The type of the i-th element is given by element_types[i].
        The description relating the geometry and the element type is given by the 'point_ordering.yaml' file.

    inpoel: numpy.ndarray
        Elements connectivity, passed by the user

    esup : numpy.ndarray
        Elements surrounding points connectivity
    esup_ptr : numpy.ndarray
        Elements surrounding points pointer. 
        i.e: The elements surrounding point i are in esup[esup_ptr[i]:esup_ptr[i+1]]

    psup : numpy.ndarray
        Points surrounding points connectivity
    psup_ptr : numpy.ndarray
        Points surrounding points pointer. 
        i.e: The points surrounding point i are in psup[psup_ptr[i]:psup_ptr[i+1]]

    nfael : numpy.ndarray
        Number of faces per element type
        i.e : The number of faces of a tetrahedron is nfael[4]

    lnofa : numpy.ndarray
        Number of points per face per element type
        i.e : The number of points in the second face of a tetrahedron is lnofa[4, 1]

    lpofa: numpy.ndarray
        Description of the faces of each element type
        i.e : The faces of a tetrahedron are in lpofa[4, :, :]

    inpofa: numpy.ndarray
        Points that compose each face
        i.e : The points that compose face i are in inpofa[i, :])

    infael: numpy.ndarray
        Faces that compose each element
        i.e : The faces that compose element i are in infael[i, :])
        
    esufa: numpy.ndarray
        Elements surrounding faces
        i.e : The elements surrounding face i are in esufa[i, :])

    nelfa: numpy.ndarray
        Number of elements that share a face
        i.e : The number of elements that share face i is nelfa[i]

    esuel : numpy.ndarray
        Elements surrounding elements 
        i.e : The elements surrounding element i are in esuel[i, :])
    
    nedel: numpy.ndarray
        Number of edges per element type
        i.e : The number of edges of element of type i is nedel[i]
    
    lpoed : numpy.ndarray
        Description of the edges of each element type
        i.e : The edges of an element of type i are in lpoed[i, :, :]

    inedel : numpy.ndarray
        Edges that compose each element
        i.e : The edges that compose element i are in edsuel[i, :])

    """
    cdef readonly int n_dims

    cdef readonly int n_elems
    cdef readonly int n_points
    cdef readonly int n_faces

    cdef readonly int are_elements_loaded
    cdef readonly int are_coords_loaded

    cdef readonly DTYPE_F_t[:, ::1] point_coords
    cdef readonly DTYPE_F_t[:, ::1] centroids
    
    cdef readonly DTYPE_I_t[::1] n_points_per_elem

    cdef readonly DTYPE_I_t[::1] element_types

    cdef readonly DTYPE_I_t[:, ::1] inpoel

    cdef readonly DTYPE_I_t[::1] esup
    cdef readonly DTYPE_I_t[::1] esup_ptr

    cdef readonly DTYPE_I_t[::1] psup
    cdef readonly DTYPE_I_t[::1] psup_ptr
    
    cdef readonly DTYPE_I_t[::1] nfael
    cdef readonly DTYPE_I_t[:, ::1] lnofa
    cdef readonly DTYPE_I_t[:, :, ::1] lpofa

    cdef readonly DTYPE_I_t[:, ::1] inpofa
    cdef readonly DTYPE_I_t[:, ::1] infael

    cdef readonly DTYPE_I_t[:, ::1] esufa
    cdef readonly DTYPE_I_t[::1] nelfa
    
    cdef readonly DTYPE_I_t[:, ::1] esuel

    cdef readonly DTYPE_I_t[::1] nedel
    cdef readonly DTYPE_I_t[:, :, ::1] lpoed

    cdef readonly DTYPE_I_t[:, ::1] inedel
    

    """
        Builds the necessary grid data structures for the given connectivity matrix, that being:
            - Elements surrounding each point   (esup)
            - Points surrounding each point     (psup)
            - Number of faces                   (n_faces)
            - Elements surrounding each element (esuel)

        In the 3D case, the following data structures are also built:
            - Points that form each edge        (inpoed)
            - Edges that form each element      (ledel)

        Parameters
        ---------
            connectivity : np.ndarray
                Connectivity matrix. Each row contains the indices of the points that form an element.
                The connectivity matrix shall be a 2D array of shape (n_elems, 8), such that
                connectivity[i, :] contains the indices of the points that form the i-th element. 

                Reference on the ordering may be found at 'utils/point_ordering.yaml'

                THE COLUMNS WITH UNUSED POINTS SHALL BE FILLED WITH -1.
            element_types : np.ndarray
                Array containing the type of each element. The type of the i-th element is given by element_types[i].
                The description relating the geometry and the element type is given by the 'point_ordering.yaml' file.
                
        Notes
        -----
        Assumes that the connectivity matrix is contiguous in memory, and that elements and points are 0-indexed in the mesh.
        If this parameter is not given, the element type will be inferred from the connectivity matrix.
            In this case, elements with 4 points will be considered as tetrahedrons

        References
        ----------
        [1] Löhner, R. (2008). Applied Computational Fluid Dynamics Techniques: 
            An Introduction Based on Finite Element Methods (2nd ed.). ISBN: 978-0-470-51907-3.

    """ 
    cpdef void build(self, const DTYPE_I_t[:, ::1] connectivity, const DTYPE_I_t[::1] element_types, const DTYPE_I_t[::1] n_points_per_elem)

    """
        Builds the elements surrounding each point (esup) array.

        Notes
        -----
        Assumes that the "inpoel" array has been calculated.
    """
    cdef void build_esup(self)

    """
        Calculates the points surrounding each point.

        Notes
        -----
        Assumes that the "esup" array has been calculated.
    """
    cdef void build_psup(self)

    cdef void build_infael(self)

    cdef void build_esufa(self)

    """
        Builds the elements surrounding each element (esuel) array. 
        Iterates over elements faces and checks if the face is shared with another element, by checking every element that surrounds a point of the face.

        Notes
        -----
        Assumes that the "inpoel" and "esup" arrays have been calculated.
        Rather convoluted implementation, perhaps this can be improved.
    """
    cdef void build_esuel(self)

    cdef void build_inedel(self)
    
    cdef void load_point_coords(self, DTYPE_F_t[:, ::1] point_coords)

    cdef void calculate_cells_centroids(self)