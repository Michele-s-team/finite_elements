'''
Ths code generates a 1d mesh given by a segment with a vertex in the segment

Run with
    clear; clear; python3 generate_mesh_line_vertex.py [path where to read the parameter file] [path where to store the solution]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/1d/line_vertex"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/1d/line_vertex/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_line_vertex.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh
import runtime_arguments_generate_mesh as rarg
import read_parameters_generate_mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file_name = output_directory + "mesh.msh"
mesh_metadata_file_name = output_directory + 'mesh_metadata.csv'

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a 1d object a set of lines
points = [model.add_point((0, 0, 0), mesh_size=rpam.parameters['resolution']),
          model.add_point((rpam.parameters["x_p"], 0, 0), mesh_size=rpam.parameters['resolution']),
          model.add_point((rpam.parameters["L"], 0, 0), mesh_size=rpam.parameters['resolution'])
          ]
my_lines = [model.add_line(points[0], points[1]), model.add_line(points[1], points[2])]

model.synchronize()

# print("# of lines added = ", len(my_lines))

model.add_physical([my_lines[0]], "line_1")
model.add_physical([my_lines[1]], "line_2")
model.add_physical([points[0]], "point_l")
model.add_physical([points[2]], "point_r")
model.add_physical([points[1]], "point_in")

geometry.generate_mesh(dim=3)
gmsh.write(mesh_file_name)

# # print line mesh to xdmf file
# msh.write_mesh_components(mesh_file_name, output_directory + "line_mesh.xdmf", "line", True)
#
# #  print vertex mesh to xdmf file
# msh.write_mesh_components(mesh_file_name, output_directory + "vertex_mesh.xdmf", "vertex", True)
#
# # print mesh vertices to csv file
# mesh = msh.read_mesh(output_directory + "line_mesh.xdmf")
# io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")
#
# # print mesh lines to csv file
# msh.print_mesh_lines_to_csv(mesh_file_name, output_directory + 'line_vertices.csv')
#
# # print mesh metadata to csv file
# io.write_parameters_to_csv_file(mesh_metadata_file_name, rpam.parameters)

msh.full_write(mesh_file_name, ['line', 'vertex'], rpam.parameters, output_directory, True)

model.__exit__()
