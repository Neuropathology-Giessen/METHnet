import os

def create_folder(folder):
        
    subfolders = folder.split('/')
    current_path = ''

    for sf in subfolders:
        current_path += sf + '/'

        if not os.path.exists(current_path):
            os.mkdir(current_path)

def create_folder_per_property(root_folder, property):
    pass