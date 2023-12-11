import os
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

directory_path = os.path.dirname(os.path.abspath(__file__)) 

project_name = 'ninpol'
ext_data = [
        Extension(
            name = f'{project_name}.grid',
            sources = [
                os.path.join(directory_path, project_name, 'grid.pyx')
            ],
            include_dirs = [
                np.get_include()
            ]
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
for e in ext_data:
    e.extra_compile_args = ['-O3', '-fopenmp']
    e.extra_link_args    = ['-fopenmp']
    e.define_macros      = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
directives = {
    'boundscheck'       : False,
    'wraparound'        : False,
    'nonecheck'         : False,
    'initializedcheck'  : False,
    'cdivision'         : True
}
setup(
    name        =  project_name,
    version     = '0.0.1',
    author      = 'Davi Yan',
    description = 'Library of Nodal Interpolation Techniques for Finite Volume Schemes',

    ext_modules = cythonize(ext_data, language_level = '3', nthreads=4, annotate=True, compiler_directives=directives, force=True)
)