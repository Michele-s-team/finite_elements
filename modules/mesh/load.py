from fenics import *
import os

import input_output as io
import mesh.utils as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(io.add_trailing_slash(rarg.args.input_directory) + "mesh_metadata.csv")


if "n_meshes" not in parameters: 
    # there is only one mesh -> read it

    # read the mesh
    mesh, sf = msh.read_from_file(rarg.args.input_directory, parameters['file_format'])

    # read the sub_meshes and generate their functions tagging cells and vertices
    sub_meshes, sf_sub_meshes, mf_sub_meshes = msh.read_sub_meshes(mesh, sf, parameters, rarg.args.input_directory)

else: 
    # there are multiple meshes -> read all of them 

    # mesh, sf and parameters store the mesh, functions, and parameters for each mesh
    mesh = [None] * parameters["n_meshes"]
    sf = [None] * parameters["n_meshes"]
    mesh_parameters = [None] * parameters["n_meshes"]

    '''
    sub_meshes, sf_sub_meshes and mf_sub_meshes store the sub_meshes and the mesh functions for each sub_mesh of the parent mesh
    Example: 
        sub_meshes[i][j] is the j-th sub_mesh of mesh[i] and sf_sub_meshes[i][j] is the mesh function of sub_meshes[i][j] 
    '''
    sub_meshes = [None] * parameters["n_meshes"]
    sf_sub_meshes = [None] * parameters["n_meshes"]
    mf_sub_meshes = [None] * parameters["n_meshes"]


    for i in range(parameters["n_meshes"]):
        # run through the meshes

        mesh_parameters[i] = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, f'mesh_{i}', "mesh_metadata.csv"))


        # read the i-th mesh
        mesh[i], sf[i] = msh.read_from_file(os.path.join(rarg.args.input_directory, f'mesh_{i}'), parameters[f'mesh_{i}_file_format'])

        # read the sub_meshes and generate their functions tagging cells and vertices
        sub_meshes[i], sf_sub_meshes[i], mf_sub_meshes[i] = msh.read_sub_meshes(mesh[i], sf[i], mesh_parameters[i], os.path.join(rarg.args.input_directory, f'mesh_{i}'))



