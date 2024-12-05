# ninpol

Library of **N**odal **In**ter**pol**ation Techniques for Finite Volume Schemes

## Installation
```bash
pip install ninpol
```

## Pre-requisites
```bash
requires = [
  "setuptools",
  "wheel",
  "Cython",
  "numpy",
  "scipy"
]
```
## Usage
```python
import ninpol

interpolator = ninpol.Interpolator(logging = True)
interpolator.load_mesh(mesh_file)

weights, neumann = interpolator.interpolate(variable, method)
```
Where:
- `mesh_file` is the path to the mesh file
- `variable` is the variable to be interpolated, associated with the elements of the mesh
- `method` is the interpolation method to be used (can be checked with `interpolator.supported_methods`)
- `weights` is a sparse matrix of shape `(n_nodes, n_elements)` containing the interpolation weights for each node. 
- `neumann` is a numpy array of shape `(n_nodes)` containing the value of the Neumann boundary condition values for each node. It's filled with `0`'s if there's no node in a Neumann boundary condition domain. 

## Tests
On linux, run, on the project root directory:
```bash
make test
```

On Windows or MacOS:
```bash
pytest -s --tb=short
python3 ./tests/results/graph.py
```

The second command will generate the graphs and `.csv` files with the results of the tests.
**OBS**: Requires `pytest` to be installed.