"""
This file contains the "Interpolator" class definition, for interpolating mesh unknows. 
"""
import numpy as np

DTYPE_I = np.int64
DTYPE_F = np.float64

cdef class Interpolator:
    def __cinit__(self):
        pass
        