import os
import shutil

directory_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(directory_path)
backup_path = os.path.join(directory_path, 'backup')

# Evil, but i'm lazy
while True:
    try:
        # Create a backup directory. If it already exists, clean it
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        else:
            # Clean the existing backup directory
            shutil.rmtree(backup_path)
            os.makedirs(backup_path)

        # Function to move files with specific extensions to the backup directory
        def move_files_by_extension(src_dir, dest_dir, extensions):
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith(extensions):
                        file_path = os.path.join(root, file)
                        dest_file_path = os.path.join(dest_dir, f"{file}_{len(os.listdir(dest_dir))}")
                        shutil.move(file_path, dest_file_path)

        # Function to move folders to the backup directory with a suffix index
        def move_folders_to_backup(src_dir, dest_dir, folder_name, suffix_index):
            for root, dirs, files in os.walk(src_dir):
                for folder in dirs:
                    if folder_name in folder:
                        folder_path = os.path.join(root, folder)
                        dest_folder_path = os.path.join(dest_dir, f"{folder}_{suffix_index}")
                        shutil.move(folder_path, dest_folder_path)

        # Move every folder that has 'cache' in its name to ./backup
        index = 0
        for root, dirs, files in os.walk(parent_path):
            for folder in dirs:
                if 'cache' in folder:
                    move_folders_to_backup(root, backup_path, folder, index)
                    index += 1

        # Move every folder with egg, x86_64 to ./backup
        for root, dirs, files in os.walk(parent_path):
            for folder in dirs:
                if 'egg' in folder or 'x86_64' in folder:
                    move_folders_to_backup(root, backup_path, folder, index)
                    index += 1
        
        # Move "dist" folder to ./backup
        for root, dirs, files in os.walk(parent_path):
            for folder in dirs:
                if 'dist' in folder:
                    move_folders_to_backup(root, backup_path, folder, index)
                    index += 1

        # Move every .so, .html, .c file to ./backup
        move_files_by_extension(parent_path, backup_path, ('.so', '.html', '.c', '.cpp'))

        print("Script completed successfully.")
        break
    except Exception as e:
        continue
