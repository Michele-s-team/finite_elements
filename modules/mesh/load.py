from fenics import *
import os

import input_output as io
import mesh.utils as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(io.add_trailing_slash(rarg.args.input_directory) + "mesh_metadata.csv")


if "n_meshes" not in parameters: 
    # there is only one mesh -> read it

    print(f'****** There is one mesh.')

    # read the mesh
    mesh, sf = msh.read_from_file(rarg.args.input_directory, parameters['file_format'])

    # read the sub_meshes and generate their functions tagging cells and vertices
    sub_meshes, cf_sub_meshes, vf_sub_meshes = msh.read_sub_meshes(mesh, sf, parameters, rarg.args.input_directory)

    '''
    if "n_sub_meshes" in parameters:

        # mesh parameters contain the field n_sub_meshes -> generate sub_meshes
        sub_meshes = []

        if parameters["n_sub_meshes"] > 1:
            # the mesh contains multiple sub_meshes: run through them and generate each sub_mesh from the parent mesh
            print('Generating sub_meshes ... ')
            for p in range(parameters["n_sub_meshes"]):

                if parameters[f'sub_mesh_{p}_dim'] > 1:

                    # the sub_mesh has dimension > 1: generate it in the ordinary way  with SubMesh
                    sub_meshes.append(SubMesh(mesh, sf, parameters[f'sub_mesh_{p}_id']))

                elif parameters[f'sub_mesh_{p}_dim'] == 1:
                    
                    # the sub_mesh has dimension 1 -> it is a line: if I generated it with 'sub_meshes.append(SubMesh(mesh, sf, parameters[f'sub_mesh_{p}_id']))' 
                    # I would obtain a one-dimensional mesh embedded in two-dimensional space, thus in fact a two-dimensional mesh, which is not what I want : I want a one-dimensional mesh. 
                    # -> I create an IntervalMesh and assign to it the coordinates of the submesh, and append to sub_meshes the IntervalMesh
                    

                    # read the line components from the parent mesh and create the relative mesh function 'cf'
                    line_mesh = msh.read_mesh(io.add_trailing_slash(rarg.args.input_directory) + "line_mesh.xdmf")
                    cf = msh.read_mesh_components(line_mesh, line_mesh.topology().dim(), io.add_trailing_slash(rarg.args.input_directory) + "line_mesh.xdmf")

                    # create  submesh_2d from the cell function 'cf' and the id which identifies the submesh: submesh_2d is a line embedded in 2d space
                    submesh_2d = SubMesh(mesh, cf, parameters[f'sub_mesh_{p}_id'])

                    # transform submesh_2d into a truly 1d mesh
                    # Extract x-coordinates from the 2D submesh
                    x_coordinates = []
                    for vertex in vertices(submesh_2d):
                        x_coordinates.append(vertex.point().x())

                    x_coordinates = sorted(list(set(x_coordinates)))  # Remove duplicates and sort

                    # generate the one-dimensional submesh and return its cell mesh function and vertex mesh function
                    sub_mesh_1d, cf_sub_mesh_1d, vf_sub_mesh_1d = msh.genereate_line_mesh(0, parameters['L'], len(x_coordinates) - 1,
                                                                                        parameters[f'sub_mesh_{p}_id'], parameters[f'vertex_sub_mesh_{p}_l_id'], parameters[f'vertex_sub_mesh_{p}_r_id'],
                                                                                        None, None)
                    sub_meshes.append(sub_mesh_1d)

            print(f'Sub_mesh {p} has dimension {sub_meshes[p].topology().dim()}')

        print('... done.')

    '''


else: 
    print(f'***** There are multiple meshes.')

    mesh = [None] * parameters["n_meshes"]
    sf = [None] * parameters["n_meshes"]

    for i in range(parameters["n_meshes"]):
        # run through the meshes

        # read the i-th mesh
        mesh[i], sf[i] = msh.read_from_file(os.path.join(rarg.args.input_directory, f'mesh_{i}'), parameters[f'mesh_{i}_file_format'])


