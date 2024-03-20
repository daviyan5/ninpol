import os
import sys
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


is_debug = False
force = False
# check if the environment variable is set
if 'DEBUG_NINPOL' in os.environ:
    is_debug = True
    force = True
directory_path = os.path.dirname(os.path.abspath(__file__)) 
n_threads = os.cpu_count()
project_name = 'ninpol'
ext_data = [
        Extension(
            name = f'{project_name}.interpolator',
            sources = [
                os.path.join(directory_path, project_name, 'interpolator.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
        ),
        Extension(
            name = f'{project_name}.grid',
            sources = [
                os.path.join(directory_path, project_name, 'grid.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
            #,define_macros=[('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
        ),
        Extension(
            name = f'{project_name}.methods.inv_dist',
            sources = [
                os.path.join(directory_path, project_name, 'methods', 'inv_dist.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
        )
]

package_data = {
    project_name: ['utils/point_ordering.yaml']
}

for e in ext_data:
    e.extra_compile_args = ['-O3', '-fopenmp']
    e.extra_link_args    = ['-fopenmp']
    e.define_macros      = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

directives = {
    'boundscheck'       : False,
    'wraparound'        : False,
    'nonecheck'         : False,
    'initializedcheck'  : False,
    'cdivision'         : True,
    'profile'           : False,
    'linetrace'         : False
    }



setup(
    name        =  project_name,
    version     = '0.1.0',
    author      = 'Davi Yan',
    description = 'Library of Nodal Interpolation Techniques for Finite Volume Schemes',
    packages    = find_packages(),
    package_data= package_data,  # Include data files
    ext_modules = cythonize(ext_data, 
                            language_level      =   '3', 
                            nthreads            =   n_threads, 
                            annotate            =   True, 
                            compiler_directives =   directives, 
                            force               =   force,
                            gdb_debug           =   is_debug),
    requires=['numpy', 'cython', 'meshio', 'pyyaml']
)