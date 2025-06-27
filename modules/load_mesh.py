from fenics import *
import os

import mesh as msh
import runtime_arguments as rarg

# read the mesh
mesh = msh.read_from_file(rarg.args.input_directory)


# look for sub_meshes
path_to_subfolder = os.path.join(rarg.args.input_directory, 'sub_meshes')

if os.path.isdir(path_to_subfolder):
    print(f"'{rarg.args.input_directory}' contains a subfolder called sub_meshes")
else:
    print(f"No subfolder named sub_meshes inside '{rarg.args.input_directory}'")