from fenics import *


import input_output as io
import mesh as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(io.add_trailing_slash(rarg.args.input_directory) + "mesh_metadata.csv")

# read the mesh
mesh, sf = msh.read_from_file(rarg.args.input_directory, parameters['file_format'])

if "n_sub_meshes" in parameters:
    #mesh parameters contain the field n_sub_meshes -> generate sub_meshes
    sub_meshes = []
    if parameters["n_sub_meshes"] > 1:
        # the mesh contains multiple sub_meshes: run through them and generate each sub_mesh from the parent mesh
        print('Generating sub_meshes ... ')
        for p in range(parameters["n_sub_meshes"]):
            sub_meshes.append(SubMesh(mesh, sf, parameters[f'sub_mesh_{p}_id']))

        print('... done.')