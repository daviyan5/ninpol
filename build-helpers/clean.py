import os
import shutil

directory_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(directory_path)
list_path = os.path.join(directory_path, 'list.txt')

# Create clean-backup directory if it doesn't exist
backup_dir = os.path.join(directory_path, 'clean-backup')
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
else:
    os.rmdir(backup_dir)
    os.makedirs(backup_dir)
# Save the name of the moved files in a list
moved_files = []

with open(list_path, 'r') as list_file:
    keep_files = list_file.readlines()
    keep_files = [f.strip() for f in keep_files]

    for root, dirs, files in os.walk(parent_path):
        if '.git' in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)

            if file_path not in keep_files:
                new_path = os.path.join(backup_dir, file)
                shutil.move(file_path, new_path)
                moved_files.append(file_path + ' moved to ' + new_path)

# Write moved files to a log file
moved_list = os.path.join(directory_path, 'moved.txt')
with open(moved_list, 'w') as moved_file:
    for file in moved_files:
        moved_file.write(file + '\n')