from fenics import *
import math

import input_output as io
import mesh as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(io.add_trailing_slash(rarg.args.input_directory) + "mesh_metadata.csv")

# read the mesh
mesh, sf = msh.read_from_file(rarg.args.input_directory, parameters['file_format'])

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
                '''
                the sub_mesh has dimension 1, and here it is supposed that it is a line: if I generated it with 'sub_meshes.append(SubMesh(mesh, sf, parameters[f'sub_mesh_{p}_id']))' 
                I would obtain a one-dimensional mesh embedded in two-dimensional space, which is not what I want 
                -> I create an IntervalMesh and assign to it the coordinates of the submesh, and append to sub_meshes the IntervalMesh
                '''

                # read the line components from the parent mesh and create the relative mesh function 'cf'
                line_mesh = msh.read_mesh(io.add_trailing_slash(rarg.args.input_directory) + "line_mesh.xdmf")
                cf = msh.read_mesh_components(line_mesh, line_mesh.topology().dim(), io.add_trailing_slash(rarg.args.input_directory) + "line_mesh.xdmf")

                # create  submesh_2d from the cell function 'cf' and the id which identifies the submesh: submesh_2d is a line embedded in 2d space
                submesh_2d = SubMesh(mesh, cf, parameters[f'sub_mesh_{p}_id'])

                # transform submesh_2d into a truly 1d mesh
                # Extract x-coordinates from the 2D submesh
                x_coords = []
                for vertex in vertices(submesh_2d):
                    x_coords.append(vertex.point().x())

                x_coords = sorted(list(set(x_coords)))  # Remove duplicates and sort

                # Create new 1D mesh
                sub_mesh_1d = IntervalMesh(len(x_coords) - 1, x_coords[0], x_coords[-1])

                # After creating sub_mesh_1d: tag its compoennts
                # tag the lines
                cf_sub_mesh_1d = MeshFunction("size_t", sub_mesh_1d, sub_mesh_1d.topology().dim())
                cf_sub_mesh_1d.set_all(parameters[f'sub_mesh_{p}_id'])  # Tag all cells with submesh ID

                # tag the vertices
                vf_sub_mesh_1d = MeshFunction("size_t", sub_mesh_1d, sub_mesh_1d.topology().dim() - 1)
                # Tag vertices based on position
                for vertex in vertices(sub_mesh_1d):
                    x = vertex.point().x()
                    if math.isclose(x, 0.0):
                        vf_sub_mesh_1d[vertex] = parameters['vertex_sub_mesh_1_l_id']
                    elif math.isclose(x, parameters['L']):
                        vf_sub_mesh_1d[vertex] = parameters['vertex_sub_mesh_1_r_id']

                sub_meshes.append(sub_mesh_1d)

        print(f'Sub_mesh {p} has dimension {sub_meshes[p].topology().dim()}')

    print('... done.')
