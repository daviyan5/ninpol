
import pyximport
pyximport.install()

import ninpol.mesh
import numpy as np
import read_msh
import pstats, cProfile

msh_file_path = "tests/test-mesh/test22.msh"
node_coords, matrix, elem_types = read_msh.read_msh_file(msh_file_path)

grid = ninpol.mesh.Grid(3, len(matrix), len(node_coords))
grid.build(matrix, elem_types)

# import os
# file_path = os.path.dirname(os.path.abspath(__file__))
# profile_path = os.path.join(file_path, "profiles/Profile.prof")
# cProfile.runctx("grid.build(matrix, elem_types)", globals(), locals(), profile_path)

# s = pstats.Stats(profile_path)
# s.strip_dirs().sort_stats("time").print_stats()
