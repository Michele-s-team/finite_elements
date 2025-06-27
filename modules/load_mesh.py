from fenics import *
import os

import mesh as msh
import runtime_arguments as rarg

# read the mesh
mesh = msh.read_from_file(rarg.args.input_directory)

# look for sub_meshes
path_to_subfolder = os.path.join(rarg.args.input_directory, 'sub_meshes')

sub_meshes = []
if os.path.isdir(path_to_subfolder):
    print(f"'{rarg.args.input_directory}' contains sub_meshes:")

    for sub_mesh_folder in os.listdir(path_to_subfolder):
        # loop through all subpaths of sub_mesh_folder, which may be file names, or anything
        full_path = os.path.join(path_to_subfolder, sub_mesh_folder)
        if os.path.isdir(full_path):
            # among the paths, consider only those that correspond to directories
            print('\t', sub_mesh_folder)
            sub_meshes.append(msh.read_from_file(full_path))

    # print(sub_meshes)

else:
    print(f"'{rarg.args.input_directory}' does not contain sub_meshes")
