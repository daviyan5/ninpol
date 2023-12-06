"""
This file contains the "Grid" class definition, for mesh manipulation. 
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
import numpy as np
cimport numpy as cnp
from cython.parallel import prange
cnp.import_array()                  # Needed to use NumPy C API

DTYPE_I = np.int64
DTYPE_F = np.float64

ctypedef cnp.int64_t DTYPE_I_t
ctypedef cnp.float64_t DTYPE_F_t

cdef class Grid:
    """
    Stores and manipulates the mesh data.
    This is a class intended to be used only from Cython, specifically from the 'interpolator.pyx' file.
    """
    cdef int n_dims                                 # Number of dimensions
    cdef int n_elems                                # Number of elements
    cdef int n_points                               # Number of points (vertices)
    cdef int n_points_per_elem                      # Number of points per element

    cdef int [:] esup                               # Elements surrounding points connectivity
    cdef int [:] esup_ptr                           # Elements surrounding points pointer. 
                                                    #   i.e: The elements surrounding point i are in esup[esup_ptr[i]:esup_ptr[i+1]]

    cdef int[:] psup                                # Points surrounding points connectivity
    cdef int[:] psup_ptr                            # Points surrounding points pointer. 
                                                    #   i.e: The points surrounding point i are in psup[psup_ptr[i]:psup_ptr[i+1]]

    def __cinit__(self, int n_dims, int n_elems, int n_points, int n_points_per_elem):
        """
        Initializes the grid.
        Parameter
        ---------
            n_dims : int
                Number of dimensions (i.e. 2 or 3)
            n_elems : int
                Number of elements in the mesh
            n_points : int
                Number of points (vertices) in the mesh
            n_points_per_elem : int
                Number of points (vertices) per element
        """
        self.n_dims             = n_dims
        self.n_elems            = n_elems
        self.n_points           = n_points
        self.n_points_per_elem  = n_points_per_elem

        # Reshape the arrays
        self.esup_ptr = np.zeros(n_points+1, dtype=DTYPE_I)
        self.psup_ptr = np.zeros(n_points+1, dtype=DTYPE_I)

    cdef void build(self, cnp.ndarray[DTYPE_I_t, ndim=2] connectivity):
        """
            Builds the Elements surrounding points connectivity (esup) and the Points surrounding points connectivity (psup) from the connectivity matrix.
            0-based indexing is assumed.
            Parameter
            ---------
                connectivity : np.ndarray
                    Connectivity matrix. Each row contains the indices of the points that form an element.
        """
        # Check that the connectivity matrix is not None and has the correct shape
        if connectivity is None:
            raise ValueError("The connectivity matrix cannot be None.")
        if connectivity.shape[0] != self.n_elems:
            raise ValueError("The number of rows in the connectivity matrix must be equal to the number of elements.")
        if connectivity.shape[1] != self.n_points_per_elem:
            raise ValueError("The number of columns in the connectivity matrix must be equal to the number of points per element.")

        # Count the number of elements surrounding each point
        self.esup_ptr = np.bincount(connectivity.ravel() + 1, minlength=self.n_points + 1)
        self.esup_ptr = np.cumsum(self.esup_ptr)
        
        cdef int[:, :] connectivity_view = connectivity
        # Fill the esup array

        self.esup = np.zeros(self.esup_ptr[self.n_points], dtype=DTYPE_I)
        for i in range(self.n_elems):
            for j in range(self.n_points_per_elem):
                self.esup[self.esup_ptr[connectivity_view[i, j]]] = i
                self.esup_ptr[connectivity_view[i, j]] += 1
        
        for i in range(self.n_points, 0, -1):
            self.esup_ptr[i] = self.esup_ptr[i-1]
        self.esup_ptr[0] = 0


    

        