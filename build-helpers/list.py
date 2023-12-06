# Creates a list.txt with the absolute name of every file in the parent directory of this script

import os

directory_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(directory_path)
list_path = os.path.join(directory_path, 'list.txt')

with open(list_path, 'w') as list_file:
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if '.git' not in os.path.join(root, file):
                list_file.write(os.path.join(root, file) + '\n')