"""
This file contains the "Grid" class definition, for mesh manipulation. 
"""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
import numpy as np
cimport numpy as cnp
from cython.parallel cimport parallel, prange
from libc.stdio cimport printf
cimport openmp

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

    cdef public int[:] esup                         # Elements surrounding points connectivity
    cdef public int[:] esup_ptr                     # Elements surrounding points pointer. 
                                                    #   i.e: The elements surrounding point i are in esup[esup_ptr[i]:esup_ptr[i+1]]

    cdef public int[:] psup                         # Points surrounding points connectivity
    cdef public int[:] psup_ptr                     # Points surrounding points pointer. 
                                                    #   i.e: The points surrounding point i are in psup[psup_ptr[i]:psup_ptr[i+1]]

    def __cinit__(self, DTYPE_I_t n_dims, DTYPE_I_t n_elems, DTYPE_I_t n_points, DTYPE_I_t n_points_per_elem):
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

        

    cpdef void build(self, DTYPE_I_t [:, :] connectivity):
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

        cdef:
            int i, j
            int tid
        # Initialize the lock

        # Reshape the arrays
        self.esup_ptr = np.zeros(self.n_points+1, dtype=np.int32)
        self.psup_ptr = np.zeros(self.n_points+1, dtype=np.int32)

         
        # Count the number of elements surrounding each point
        
        for i in range(self.n_elems):
            for j in range(self.n_points_per_elem):
                self.esup_ptr[connectivity[i, j] + 1] += 1
        
        # Compute the cumulative sum of the number of elements surrounding each point
        for i in range(self.n_points):
            self.esup_ptr[i + 1] += self.esup_ptr[i]

        # Fill the esup array
        self.esup = np.zeros(self.esup_ptr[self.n_points], dtype=np.int32)
        for i in range(self.n_elems):
            for j in range(self.n_points_per_elem):
                self.esup[self.esup_ptr[connectivity[i, j]]] = i
                self.esup_ptr[connectivity[i, j]] += 1
        
        for i in range(self.n_points, 0, -1):
            self.esup_ptr[i] = self.esup_ptr[i-1]
        self.esup_ptr[0] = 0

        cdef:
            int stor_ptr = 0, point_idx
            int[:] temp_psup = np.ones(self.n_points, dtype=np.int32) * -1
        self.psup_ptr[0] = 0
    
        # Upper bound for the number of points surrounding each point
        self.psup = np.zeros((self.esup_ptr[self.n_points] * (self.n_points_per_elem - 1)), dtype=np.int32) # Peharps this can be improved

        # Calculate the points surrounding each point, using temp_psup to avoid duplicates
        for i in range(self.n_points):
            for j in range(self.esup_ptr[i], self.esup_ptr[i+1]):
                for k in range(self.n_points_per_elem):
                    point_idx = connectivity[self.esup[j], k]
                    if point_idx != i and temp_psup[point_idx] != i:
                        self.psup[stor_ptr] = point_idx
                        temp_psup[point_idx] = i
                        stor_ptr += 1
                        
            self.psup_ptr[i+1] = stor_ptr
        self.psup = self.psup[:stor_ptr]



    

        