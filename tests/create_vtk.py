import meshio
import numpy as np
import ninpol
import matplotlib.pyplot as plt

def get_random_matrix(shape):
    # Return list of lists 
    if len(shape) == 1:
        if shape[0] == 1:
            rng = np.random.default_rng()
            return rng.random()
        return [np.random.rand() for _ in range(shape[0])]
    return np.random.rand(*shape).tolist()

def calculate_centroid(cell, point_coords):
    coords = point_coords[cell]
    return np.mean(coords, axis=0)

def linear_function(x):
    return 2*x[0] + 3*x[1] + 3

# two triangles and one quad
points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
]
n_points = len(points)
cells = [
    ("triangle", [[0, 1, 2], [1, 3, 2]]),
    ("quad", [[1, 4, 5, 3]]),
]

centroids = []
for cell_type, cell in cells:
    for elem in cell:
        centroids.append(calculate_centroid(elem, np.array(points)))
centroids = np.array(centroids)
print(centroids)
print(points)
cell_data = {
    "lfunc" : [
        [linear_function(centroids[0]), linear_function(centroids[1]) ],
        [linear_function(centroids[2])]
    ]
}
mesh = meshio.Mesh(
    points,
    cells,
    cell_data=cell_data,
    
)

mesh.write(
    "foo.vtk",  # str, os.PathLike, or buffer/open file
)


inter = ninpol.Interpolator()

inter.load_mesh("foo.vtk")

points = np.array(points)
interpolated_data = inter.interpolate(np.arange(n_points), 'lfunc', 'linear')

# Side plot with the linear function for every value between 0,0 and 2,1
plt.figure()
plt.title('Linear Function Values')
x = np.linspace(0, 2, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = linear_function([X, Y])
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()


plt.show()

for cell_type, cell in cells:
    for i, elem in enumerate(cell):
        # Connect the points counter-clockwise
        plt.plot(points[elem + [elem[0]], 0], points[elem + [elem[0]], 1], 'o-', color='black')
        # Cibbect the points to the centroid
        plt.plot([centroids[i, 0], points[elem[0], 0]], [centroids[i, 1], points[elem[0], 1]], 'o-', color='blue')
plt.plot(points[:, 0], points[:, 1], 'o', color='red')
# Add a text with the value of the interpolation and the true value
for i, point in enumerate(points):
    plt.text(point[0], point[1], f"{i}: {interpolated_data[i]} - {linear_function(point)}")
plt.plot(centroids[:, 0], centroids[:, 1], 'o', color='blue')
# add a text with the value 
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], f"{i}: {linear_function(centroid)}")

plt.show()

