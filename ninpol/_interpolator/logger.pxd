cdef class Logger:
    cdef readonly str directory
    cdef readonly str filename
    cdef readonly str json_filename

    cdef readonly dict data
    cdef int last_index
    
