import os
import sys
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


is_debug = False
force    = False

# check if the --debug flag has been passed
if '--debug' in sys.argv:
    is_debug = True
    force    = True

print(" ============= Debug mode:", str(is_debug) + " ============= ")

directory_path = os.path.dirname(os.path.abspath(__file__)) 
n_threads = os.cpu_count() if not is_debug else 1
project_name = 'ninpol'
ext_data = [
        Extension(
            name = f'{project_name}._interpolator.interpolator',
            sources = [
                os.path.join(directory_path, project_name, '_interpolator', 'interpolator.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
        ),
        Extension(
            name = f'{project_name}._interpolator.grid',
            sources = [
                os.path.join(directory_path, project_name, '_interpolator', 'grid.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
        ),
        Extension(
            name = f'{project_name}._methods.inv_dist',
            sources = [
                os.path.join(directory_path, project_name, '_methods', 'inv_dist.pyx')
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
    e.extra_compile_args = ['-O3', '-fopenmp'] if not is_debug else ['-O0', '-g', '-fopenmp']
    e.extra_link_args    = ['-fopenmp']
    e.define_macros      = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if is_debug:
        e.define_macros.append(('CYTHON_TRACE_NOGIL', '1'))

directives = {
    'boundscheck'       : False if not is_debug else True,
    'wraparound'        : False, # Always false, since it can hide bugs
    'nonecheck'         : False if not is_debug else True,
    'initializedcheck'  : False if not is_debug else True,
    'cdivision'         : True if not is_debug else False,
    'profile'           : False if not is_debug else True,
    'linetrace'         : False if not is_debug else True
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