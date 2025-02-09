"""
This file contains the "PseudoGrid" class implementation
"""
import numpy as np

DTYPE_I = np.int64
DTYPE_F = np.float64

cdef class PseudoGrid:
    def __cinit__(self, dict data_target, data_source):
        pass