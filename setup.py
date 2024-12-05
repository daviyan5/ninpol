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
    force    = False

print(" ============= Debug mode:", str(is_debug) + " ============= ")

directory_path = os.path.dirname(os.path.abspath(__file__)) 
n_threads = os.cpu_count() if not is_debug else 1
project_name = 'ninpol'
ext_data = [
        Extension(
            name = f'{project_name}._interpolator.interpolator',
            sources = [
                os.path.join(project_name, '_interpolator', 'interpolator.pyx')
            ]
        ),
        Extension(
            name = f'{project_name}._interpolator.grid',
            sources = [
                os.path.join(project_name, '_interpolator', 'grid.pyx')
            ],
            language='c++'
        ),
        Extension(
            name = f'{project_name}._methods.idw',
            sources = [
                os.path.join(project_name, '_methods', 'idw.pyx')
            ]
        ),
        Extension(
            name = f'{project_name}._methods.gls',
            sources = [
                os.path.join(project_name, '_methods', 'gls.pyx')
            ],
            language='c++'
        ),
        Extension(
            name = f'{project_name}._methods.ls',
            sources = [
                os.path.join(project_name, '_methods', 'ls.pyx')
            ]
        ),
        Extension(
            name = f'{project_name}._interpolator.logger',
            sources = [
                os.path.join(project_name, '_interpolator', 'logger.pyx')
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
    e.include_dirs       = [np.get_include(), os.path.join(project_name, 'utils')]
    if is_debug:
        e.define_macros.append(('CYTHON_TRACE_NOGIL', '1'))
        e.define_macros.append(('CYTHON_TRACE', '1'))
        e.define_macros.append(('CYTHON_REFNANNY', '1'))

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
    requires=['numpy', 'scipy', 'cython', 'meshio', 'pyyaml']
)