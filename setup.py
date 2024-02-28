import os
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

directory_path = os.path.dirname(os.path.abspath(__file__)) 
n_threads = os.cpu_count()
project_name = 'ninpol'
ext_data = [
        Extension(
            name = f'{project_name}.__init__',
            sources = [
                os.path.join(directory_path, project_name, '__init__.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
        ),
        Extension(
            name = f'{project_name}.mesh.__init__',
            sources = [
                os.path.join(directory_path, project_name, 'mesh', '__init__.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
        ),
        Extension(
            name = f'{project_name}.mesh.grid',
            sources = [
                os.path.join(directory_path, project_name, 'mesh', 'grid.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
            #,define_macros=[('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
        ),
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
            name = f'{project_name}.methods.linear',
            sources = [
                os.path.join(directory_path, project_name, 'methods', 'linear.pyx')
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
    version     = '0.0.2',
    author      = 'Davi Yan',
    description = 'Library of Nodal Interpolation Techniques for Finite Volume Schemes',
    packages=find_packages(),
    package_data=package_data,  # Include data files
    ext_modules = cythonize(ext_data, 
                            language_level      =   '3', 
                            nthreads            =   n_threads, 
                            annotate            =   True, 
                            compiler_directives =   directives, 
                            force               =   False),
    requires=['numpy', 'cython', 'meshio', 'pyyaml']
)