'''
generate a mesh given by a square with a polygon-shaped hole in it

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import meshio
import numpy as np
import os
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')



# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file = output_directory + "mesh.msh"

print(f'output_directory = "{output_directory}"')

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'



# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

square_points = [model.add_point((0, 0, 0), mesh_size=rpam.parameters["resolution"]),
                model.add_point((rpam.parameters["L"], 0, 0), mesh_size=rpam.parameters["resolution"]),
                model.add_point((rpam.parameters["L"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
                model.add_point((0, rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"])]

# Add lines between all points creating the rectangle
square_lines = [model.add_line(square_points[i], square_points[i + 1])
                 for i in range(-1, len(square_points) - 1)]

square_loop = model.add_curve_loop(square_lines)


polygon_coordinates = [[0.1, 0.4], [0.15, 0.3], [0.2, 0.3], [0.3, 0.2]]
polygon_points = [model.add_point((polygon_coordinates[0][0], polygon_coordinates[0][1], 0), mesh_size=rpam.parameters["resolution"])]
model.synchronize()

polygon_lines = []

for i in range(1, len(polygon_coordinates)):

    polygon_points.append(model.add_point((polygon_coordinates[i][0], polygon_coordinates[i][1], 0), mesh_size=rpam.parameters["resolution"]))
    model.synchronize()

    polygon_lines.append(model.add_line(polygon_points[i-1], polygon_points[i]))
    model.synchronize()

polygon_lines.append(model.add_line(polygon_points[-1], polygon_points[0]))
model.synchronize()

polygon_loop = model.add_curve_loop(polygon_lines)
model.synchronize()

plane_surface = model.add_plane_surface(square_loop, holes=[polygon_loop])
model.synchronize()

'''
model.add_physical([plane_surface], "Volume")
model.add_physical([square_lines[0]], "i")
model.add_physical([square_lines[2]], "o")
model.add_physical([square_lines[3]], "t")
model.add_physical([square_lines[1]], "b")
model.add_physical(polygon_loop.curves, "c")

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

msh.print_mesh_lines_to_csv(mesh_file, output_directory + 'line_vertices.csv')


mesh_from_file = meshio.read(mesh_file)
#
# line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
# meshio.write(output_directory + "line_mesh.xdmf", line_mesh)
#
# triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
# meshio.write(output_directory + "triangle_mesh.xdmf", triangle_mesh)
#
# # print the mesh vertices to file
# mesh = msh.read_mesh(output_directory + "triangle_mesh.xdmf")
# io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")

msh.full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

# print the boundary points of the boundaries given by the ellipse, where the ellipse id is 6
ellipse_id = 6
msh.sorted_boundary_points(
    msh.read_mesh(os.path.join(output_directory, 'triangle_mesh.xdmf')), 
    output_directory, 
    [ellipse_id],
    os.path.join(output_directory, 'boundary_points_id_' + str(ellipse_id) + '.csv'))


gmsh.clear()
geometry.__exit__()
'''