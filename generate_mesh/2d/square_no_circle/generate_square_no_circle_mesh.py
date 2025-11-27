'''
generate a square mesh

Run it with
    python3 generate_square_no_circle_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_no_circle_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import meshio
import gmsh
import os
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'

mesh_file = output_directory + "mesh.msh"

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

my_points = [model.add_point((0, 0, 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["L"], 0, 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((rpam.parameters["L"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
             model.add_point((0, rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"])]

# Add lines between all points creating the rectangle
channel_lines = [model.add_line(my_points[i], my_points[i + 1])
                 for i in range(-1, len(my_points) - 1)]

channel_loop = model.add_curve_loop(channel_lines)

plane_surface = model.add_plane_surface(channel_loop, holes=[])

model.synchronize()

model.add_physical([plane_surface], "Volume")
model.add_physical([channel_lines[0]], 'l')
model.add_physical([channel_lines[2]], 'r')
model.add_physical([channel_lines[3]], 't')
model.add_physical([channel_lines[1]], 'b')

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

msh.print_mesh_lines_to_csv(mesh_file, output_directory + 'line_vertices.csv')

mesh_from_file = meshio.read(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

# print the boundary points of the boudary given by the square
# this is a list that contains the IDs of the lines: left, right, top, bottom
line_lrtb_id = [2, 3, 4, 5]
# print the boundary points which belong to edges which are identified by line_lrtb_id
msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory, 'triangle_mesh.xdmf')), 
    output_directory, 
    line_lrtb_id,
    os.path.join(output_directory, 'boundary_points_id_' + str(line_lrtb_id) + '.csv'))


gmsh.clear()
geometry.__exit__()
