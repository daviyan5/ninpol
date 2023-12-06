# Cleans every file in the parent directory that's not in list neither in .git folder

import os

directory_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(directory_path)
list_path = os.path.join(directory_path, 'list.txt')
# Save the name of the deleted files in a list
deleted_files = []
with open(list_path, 'r') as list_file:
    keep_files = list_file.readlines()
    for i in range(len(keep_files)):
        keep_files[i] = keep_files[i].strip()
    
    for root, dirs, files in os.walk(parent_path):
        if '.git' in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            
            if file_path not in keep_files:
                deleted_files.append(file_path)
                print('Deleting ' + file_path)
                os.remove(file_path)

    for root, dirs, files in os.walk(parent_path):
        if '.git' in root:
            continue
        # If the directory is empty, delete it
        if not os.listdir(root) and root != parent_path:
            print('Deleting ' + root)
            os.rmdir(root)
        
deleted_list = os.path.join(directory_path, 'deleted.txt')
with open(deleted_list, 'w') as deleted_file:
    for file in deleted_files:
        deleted_file.write(file + '\n')