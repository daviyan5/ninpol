import pytest
import pytest_subtests
import numpy as np
import meshio
import ninpol
import os
import sys
import yaml
from colorama import Fore, Style



# Test parameters
mesh_dir    = "tests/utils/altered_mesh/"
output_dir  = "tests/utils/result_mesh/"
n_files     = -1

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def get_centroid(eltype, points, faces, edges, coords, point_ordering):
    if eltype == 0:
        return coords
    simplexes = ["vertex", "line", "triangle", "tetra"]
    if eltype in [point_ordering[s]["element_type"] for s in simplexes]:
        return np.mean(coords, axis = 0)

    polygon = ["quad"]
    if eltype in [point_ordering[s]["element_type"] for s in polygon]:
        # Triangulate
        geometric_center = np.mean(coords, axis=0)
        centroid = np.array([0., 0., 0.])
        edge_coords_sum = coords[edges].sum(axis=1)
        centroid = np.mean((1/3) * (edge_coords_sum + geometric_center) , axis=0)

        return centroid
    
    centroid = np.array([0., 0., 0.])

    # 1 - Calculate the geometric centre of the polyhedron by summing all the vertices(points) and dividing by their total number
    geometric_center = np.mean(coords, axis=0)
    
    # 2 - Divide the polyhedron into a set of sub-pyramids, with the geometric centre as the apex with each surface polygon as the base.
    for i in range(len(faces)):
        faces_coords = coords[faces[i]]
        geometric_center_pyramid = np.mean(faces_coords, axis=0)
        centroid_pyrimid = (3/4) * geometric_center + (1/4) * geometric_center_pyramid
        centroid += centroid_pyrimid / len(faces)
    
    return centroid
    
# Tests wheter the grid and it's structure's have been built correctly
class TestCorrectness:
    

    def test_structures(self, subtests):
        global n_files
        prefixes = ["box", "hexa", "prism", "spheres", "tetra"]
        files   = sorted(os.listdir(mesh_dir))
        # For each prefix, keep only one file of each prefix
        # This is done because it's unlikely to have a different result between two meshes that differ only in refinement
        temp_files = []
        for file in files:
            for prefix in prefixes:
                if prefix in file:
                    prefixes.remove(prefix)
                    temp_files.append(file)
                    break
        files = temp_files
        
        # Remove .gitkeep
        if ".gitkeep" in files:
            files.remove(".gitkeep")
        
        if n_files == -1:
            n_files = len(files)
        
        
        results = {}
        results["n_files"] = n_files
        results["files"] = {}

        point_ordering_path = os.path.join(os.path.dirname(__file__), "..", "ninpol", "utils", "point_ordering.yaml")
        point_ordering = yaml.safe_load(open(point_ordering_path))
        por_idx = {}
        for key, value in point_ordering["elements"].items():
            por_idx[value["element_type"]] = value
            por_idx[value["element_type"]]["name"] = key

        TOTAL_TESTS = 10
        print("\n=======================================================================================")
        print(f"{Fore.WHITE}{'':<13}{'File':<15}{'Nº Points':<15}{'Nº Elements':<15}{'Nº Faces':<10}{'Nº Edges':<10}{'Nº Passed'}{Style.RESET_ALL}")
        for case in range(n_files):
            n_passed = 0
            interpolador = ninpol.Interpolator()
            interpolador.load_mesh(mesh_dir + files[case])
            
            n_points    = interpolador.grid_obj.n_points
            n_elements  = interpolador.grid_obj.n_elems

            n_faces     = interpolador.grid_obj.n_faces
            n_edges     = interpolador.grid_obj.n_edges

            # Test size of each array
            data = interpolador.grid_obj.get_data()

            inpoel        = data["inpoel"]         # Points that compose each element
            element_types = data["element_types"]  # Each element type
            inpofa        = data["inpofa"]         # Points that compose each face
            infael        = data["infael"]         # Faces that compose each element
            inpoed        = data["inpoed"]         # Points that compose each edge
            inedel        = data["inedel"]         # Edges that compose each element
            esup          = data["esup"]           # Elements that compose each element
            psup          = data["psup"]           # Points that compose each point
            esufa         = data["esufa"]          # Faces that compose each element



            points_coords = data["point_coords"]   # Coordinates of each point
            centroids     = data["centroids"]      # Centroids of each element

            with subtests.test('Test "Element Types"'):
                # For every element in inpoel, the number of values != -1 in the element's row should be equal to the number of points of the element_type
                n_points_by_element_type = np.array([por_idx[element_type]["number_of_points"] for element_type in element_types])
                count = np.count_nonzero(inpoel != -1, axis=1)
                assert np.all(count == n_points_by_element_type), "Element types don't match the number of points in the element"
                n_passed += 1

            with subtests.test('Test "Infael"'):
                # For every element in infael, the number of values != -1 in the element's row should be equal to the number of faces of the element_type
                n_faces_by_element_type = np.array([len(por_idx[element_type]["faces"]) for element_type in element_types])
                count = np.count_nonzero(infael != -1, axis=1)
                assert np.all(count == n_faces_by_element_type), "Element types don't match the number of faces in the element"
                n_passed += 1

            with subtests.test('Test "Inpofa"'):
                # The union of every point that composes the faces (inpofa) of an element (infael) should be equal to the points that compose the element (inpoel)
                # For every element, the faces are defined in point_ordering. That is, for the respective element type,
                #   point_ordering[type]["faces"] contains the points that compose the faces of the element respective to that element point ordering (inpoel)
                # That means that, for every row in inpoel, row[faces] = inpofa[infael_row]

                n_points_by_face = np.count_nonzero(inpofa != -1, axis=1)
                
                for i in range(inpoel.shape[0]):
                    faces_def = np.array(por_idx[element_types[i]]["faces"], dtype=object)

                    if len(faces_def) == 0:
                        continue

                    faces_def = np.concatenate(faces_def).flatten().astype(int)
                    # [1, 3, 4, 5...] -> Global index of the points that compose the faces of the element, in order
                    faces_conv = inpoel[i, faces_def] 

                    faces_el = infael[i, infael[i] >= 0]
                    faces_el_conv = []
                    for face in faces_el:
                        faces_el_conv.extend(inpofa[face, :n_points_by_face[face]])
                    faces_el_conv = np.array(faces_el_conv)
                    faces_el_conv_sorted = np.array(faces_el_conv).sort()
                    faces_conv_sorted = np.array(faces_conv).sort()
                    assert np.all(faces_el_conv_sorted == faces_conv_sorted), f"Inpofa error at element {i} of type {por_idx[element_types[i]]['name']}"

                n_passed += 1

            with subtests.test('Test "Inedel"'):
                # For every element in Inedel, the number of values != -1 in the element's row should be equal to the number of edges of the element_type
                n_edges_by_element_type = np.array([len(por_idx[element_type]["edges"]) for element_type in element_types])
                count = np.count_nonzero(inedel != -1, axis=1)
                assert np.all(count == n_edges_by_element_type), "Element types don't match the number of edges in the element"
                
                n_passed += 1
            with subtests.test('Test "Inpoed"'):
                # The union of every point that composes the edges (inpoed) of an element (inedel) should be equal to the points that compose the element (inpoel)
                # For every element, the edges are defined in point_ordering. That is, for the respective element type,
                #   point_ordering[type]["edges"] contains the points that compose the edges of the element respective to that element point ordering (inpoel)
                # That means that, for every row in inpoel, row[edges] = inpoed[inedel_row]

                n_points_by_edge = np.count_nonzero(inpoed != -1, axis=1)

                for i in range(inpoel.shape[0]):
                    edges_def = np.array(por_idx[element_types[i]]["edges"]).flatten().astype(int)

                    # [1, 3, 4, 5...] -> Global index of the points that compose the edges of the element, in order
                    edges_conv = inpoel[i, edges_def] 

                    edges_el = inedel[i, inedel[i] >= 0]
                    edges_el_conv = []
                    for edge in edges_el:
                        edges_el_conv.extend(inpoed[edge, :n_points_by_edge[edge]])
                    edges_el_conv_sorted = np.array(edges_el_conv).sort()
                    edges_conv_sorted = np.array(edges_conv).sort()
                    
                    assert np.all(edges_el_conv_sorted == edges_conv_sorted), f"Inpoed error at element {i} of type {por_idx[element_types[i]]['name']}"
                    
                n_passed += 1

            with subtests.test('Test "Esup"'):
                # For every element in esup[i], the point i should be in inpoel[esup[i]].

                for i in range(esup.shape[0]):
                    esup_nn = esup[i][esup[i] >= 0]
                    freq = np.count_nonzero(inpoel[esup_nn].flatten() == i)
                    assert freq == esup_nn.shape[0], "Elements surrounding a point don't match the points surrounding the element"
                            
                n_passed += 1
            with subtests.test('Test "Psup"'):
                # For every point p in psup[i]:
                # - There must be at least one element in esup[i] that contains p 
                # - There must be at least one element in esup[p] that contains i
                for i in range(psup.shape[0]):
                    psup_nn = psup[i][psup[i] >= 0]
                    esup_nn_i = esup[i][esup[i] >= 0]
                    for p in psup_nn:
                        esup_nn_p = esup[p][esup[p] >= 0]
                        assert np.any(inpoel[esup_nn_i] == p) and np.any(inpoel[esup_nn_p] == i), "Points surrounding a point don't match the elements surrounding the point"
                n_passed += 1
            with subtests.test('Test "Esufa"'):
                # For every element in esufa[i], the face i should be in infael[esufa[i]].
                for i in range(esufa.shape[0]):
                    esufa_nn = esufa[i][esufa[i] >= 0]
                    freq = np.count_nonzero(infael[esufa_nn].flatten() == i)
                    assert freq == esufa_nn.shape[0], "Faces surrounding an element don't match the elements surrounding the face"
                n_passed += 1
            with subtests.test('Test "Centroids"'):
                # For every element, the centroid should be the center of mass
                for i in range(inpoel.shape[0]):
                    inpoel_nn = inpoel[i][inpoel[i] >= 0]
                    inpofa_nn = por_idx[element_types[i]]["faces"]
                    inpoed_nn = por_idx[element_types[i]]["edges"]
                    
                    centroid = get_centroid(element_types[i], 
                                            inpoel_nn, inpofa_nn, inpoed_nn,
                                            points_coords[inpoel_nn], point_ordering["elements"])
                    err = np.linalg.norm(centroid - centroids[i])
                    
                    assert np.allclose(centroid, centroids[i]), f"Centroid of an element doesn't match, err: {err}"

                n_passed += 1
            with subtests.test('Test "Weights" for Distance Inverse'):
                # Interpolate 'linear' for all points and receive the weights matrix
                weights, connectivity_vals = interpolador.interpolate(np.arange(n_points), "linear", "inv_dist")

                # For every point, the sum of the weights should be 1
                for i in range(n_points):
                    assert np.isclose(np.sum(weights[i]), 1), "The sum of the weights should be 1"

                # For every point, for every elem in esup of point 
                #   connectivity_vals[point, j] = linear[esup[point, j]]
                linear = interpolador.get_data("cells", np.arange(n_elements), "linear")

                for i in range(n_points):
                    for j in range(esup[i].shape[0]):
                        if esup[i, j] == -1:
                            continue
                        assert np.isclose(connectivity_vals[i, j], linear[esup[i, j]]), "Connectivity values don't match the linear interpolation"

                # For every point, for every elem in esup of point
                #   weights[point, j] = 1 / distance(point, centroid(elem)) / total_distance

                for i in range(n_points):
                    total_dist = 0
                    for j in range(esup[i].shape[0]):
                        if esup[i, j] == -1:
                            continue
                        elem = esup[i, j]
                        dist = np.linalg.norm(points_coords[i] - centroids[elem])
                        if np.isclose(dist, np.finfo(float).eps):
                            # Weight must be 1 if the distance is 0, and 0 everywhere else
                            is_one = np.isclose(weights[i, j], 1)
                            is_zero = np.allclose(weights[i, :j], 0) and np.allclose(weights[i, j+1:], 0)

                            assert is_one and is_zero, "Weights don't match the distance inverse interpolation"
                            found_zero = True
                            break
                        total_dist += dist
                    if found_zero:
                        continue
                    for j in range(esup[i].shape[0]):
                        if esup[i, j] == -1:
                            continue
                        elem = esup[i, j]
                        dist = np.linalg.norm(points_coords[i] - centroids[elem])
                        assert np.isclose(weights[i, j], 1 / (total_dist * dist)), "Weights don't match the distance inverse interpolation"

                n_passed += 1
                
            pstr = (f"{Fore.WHITE}{'':<4}{files[case]:<15}{Fore.BLUE}" + 
                    f"{n_points:<15}{n_elements:<15}{n_faces:<10}{n_edges:<10}" +
                    f"{Fore.GREEN if n_passed == TOTAL_TESTS else Fore.RED}{n_passed:<10}{Style.RESET_ALL}")
            
            print(pstr)