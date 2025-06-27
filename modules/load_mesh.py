from fenics import *
# import os

import input_output as io
import mesh as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")


# read the mesh
mesh, sf = msh.read_from_file(rarg.args.input_directory)



# generate sub_meshes
sub_meshes = []
if parameters["n_sub_meshes"] > 1:
    print('generating sub_meshes')
    for i in range(parameters["n_sub_meshes"]):
        sub_meshes.append(SubMesh(mesh, sf, parameters[f'submesh_{i}_id']))




# path_to_subfolder = os.path.join(rarg.args.input_directory, 'sub_meshes')
#
# sub_meshes = []
# if os.path.isdir(path_to_subfolder):
#     print(f"'{rarg.args.input_directory}' contains sub_meshes:")
#
#     for sub_mesh_folder in os.listdir(path_to_subfolder):
#         # loop through all subpaths of sub_mesh_folder, which may be file names, or anything
#         full_path = os.path.join(path_to_subfolder, sub_mesh_folder)
#         if os.path.isdir(full_path):
#             # among the paths, consider only those that correspond to directories
#             print('\t', sub_mesh_folder)
#             sub_meshes.append(msh.read_from_file(full_path))
#
#     # print(sub_meshes)
#
# else:
#     print(f"'{rarg.args.input_directory}' does not contain sub_meshes")
