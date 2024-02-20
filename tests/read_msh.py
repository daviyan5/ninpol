import numpy as np

def read_msh_file(file_path):

    node_coords = {}
    elem_connectivity = {}
    elem_types = {}

    with open(file_path, 'r') as file:
       # look for $Nodes
        line = file.readline()
        while line.find("$Nodes") == -1:
            line = file.readline()

        is_old_file = False
        # next line has  numEntityBlocks(size_t) numNodes(size_t) minNodeTag(size_t) maxNodeTag(size_t)
        # We want only the number of nodes
        line = file.readline()
        if len(line.split()) != 4:
            num_nodes = int(line.split()[0])
            is_old_file = True
        else:
            num_nodes = int(line.split()[1])
        # Now, every line is of type entityDim(int) entityTag(int) parametric(int; 0 or 1) numNodesInBlock(size_t)
        # We want to read only numNodesInBlock
        line = file.readline()
        while line.find("$EndNodes") == -1:
            if is_old_file:
                num_nodes_in_block = num_nodes
                node_ids = []
                for i in range(num_nodes_in_block):
                    
                    split_line = line.split()
                    node_ids.append(int(split_line[0]))
                    node_coords[node_ids[i]] = [float(x) for x in split_line[1:]]
                    line = file.readline()

            else:
                num_nodes_in_block = int(line.split()[3])
            
                node_ids = []
                for i in range(num_nodes_in_block):
                    line = file.readline()
                    node_ids.append(int(line.split()[0]))
                for i in range(num_nodes_in_block):
                    node_coords[node_ids[i]] = [float(x) for x in file.readline().split()]
                line = file.readline()

       # look for $Elements
        while line.find("$Elements") == -1:
            line = file.readline()
        
        # Next line has numEntityBlocks(size_t) numElements(size_t) minElementTag(size_t) maxElementTag(size_t)
        # We want only the number of elements
        line = file.readline()
        if is_old_file:
            num_elements = int(line.split()[0])
        else:
            num_elements = int(line.split()[1])
        line = file.readline()

        # Next line has entityDim(int) entityTag(int) elementType(int; see below) numElementsInBlock(size_t)
        # We want to read only numElementsInBlock
        while line.find("$EndElements") == -1:
            if is_old_file:
                for i in range(num_elements):
                    split_line = line.split()
                    elem_id = int(split_line[0])
                    elem_type = int(split_line[1])
                    elem_connectivity[elem_id] = [int(x) for x in split_line[5:]]
                    elem_types[elem_id] = elem_type
                    line = file.readline()
                break
            
            else:
                element_type = int(line.split()[2])
                num_elements_in_block = int(line.split()[3])

                for i in range(num_elements_in_block):
                    line = file.readline()
                    split_line = line.split()
                    elem_id = int(split_line[0])
                    elem_connectivity[elem_id] = [int(x) for x in split_line[1:]]
                    elem_types[elem_id] = element_type
            line = file.readline()

    # Convert to numpy arrays
    node_coords_np = np.zeros((num_nodes, 3))
    
    # Convert every node key into a index 
    node_index = {}
    for i, (key, value) in enumerate(node_coords.items()):
        node_index[key] = i

    # Convert every element key into a index
    elem_index = {}
    for i, key in enumerate(elem_connectivity.keys()):
        elem_index[key] = i
    
    for key, value in node_coords.items():
        node_coords_np[node_index[key]] = value
    
    elem_connectivity_np = np.ones((num_elements, 8), dtype=int) * -1
    
    for key, value in elem_connectivity.items():
        new_value = [node_index[x] for x in value]
        elem_connectivity_np[elem_index[key], :len(value)] = new_value
    
    elem_types_np = np.zeros(num_elements, dtype=int)
    for key, value in elem_types.items():
        elem_types_np[elem_index[key]] = value

    return node_coords_np, elem_connectivity_np, elem_types_np

def main():
    # Example usage:
    msh_file_path = "tests/test-mesh/test1.msh"
    node_coords, elem_connectivity, elem_types = read_msh_file(msh_file_path)
    print(node_coords)
    print(elem_connectivity)
    print(elem_types)
    

if __name__ == '__main__':
    main()