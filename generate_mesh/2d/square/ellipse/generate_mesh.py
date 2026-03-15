'''
generate a mesh given by a square with a ellipse-shaped hole in it: the hole has the shape of an ellipse

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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


# left focal point  of the ellipse
focus = np.subtract(rpam.parameters["c"], [np.sqrt(rpam.parameters["a"] ** 2 - rpam.parameters["b"] ** 2), 0, 0])

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

p_ellipse_c = model.add_point(
    np.add(focus, np.dot(cal.R_z(rpam.parameters["phi"]), np.subtract(rpam.parameters["c"], focus)))
    , mesh_size=rpam.parameters["resolution"])
p_ellipse_r = model.add_point(
    np.add(focus, np.dot(cal.R_z(rpam.parameters["phi"]), np.subtract(np.add(rpam.parameters["c"], [rpam.parameters["a"], 0, 0]), focus))),
    mesh_size=rpam.parameters["resolution"])
p_ellipse_t = model.add_point(
    np.add(focus, np.dot(cal.R_z(rpam.parameters["phi"]), np.subtract(np.add(rpam.parameters["c"], [0, rpam.parameters["b"], 0]), focus))),
    mesh_size=rpam.parameters["resolution"])
p_ellipse_l = model.add_point(
    np.add(focus, np.dot(cal.R_z(rpam.parameters["phi"]), np.subtract(np.subtract(rpam.parameters["c"], [rpam.parameters["a"], 0, 0]), focus))),
    mesh_size=rpam.parameters["resolution"])
p_ellipse_b = model.add_point(
    np.add(focus, np.dot(cal.R_z(rpam.parameters["phi"]), np.subtract(np.subtract(rpam.parameters["c"], [0, rpam.parameters["b"], 0]), focus))),
    mesh_size=rpam.parameters["resolution"])
# p_ellipse_focus = model.add_point(focus, mesh_size=rpam.parameters["resolution"])

model.synchronize()

ellipse_arc_rt = model.add_ellipse_arc(p_ellipse_r, p_ellipse_c, p_ellipse_r, p_ellipse_t)
ellipse_arc_tl = model.add_ellipse_arc(p_ellipse_t, p_ellipse_c, p_ellipse_r, p_ellipse_l)
ellipse_arc_lb = model.add_ellipse_arc(p_ellipse_l, p_ellipse_c, p_ellipse_r, p_ellipse_b)
ellipse_arc_br = model.add_ellipse_arc(p_ellipse_b, p_ellipse_c, p_ellipse_r, p_ellipse_r)
model.synchronize()

ellipse_lines = [ellipse_arc_rt, ellipse_arc_tl, ellipse_arc_lb, ellipse_arc_br]
ellipse_loop = model.add_curve_loop(ellipse_lines)
model.synchronize()

plane_surface = model.add_plane_surface(channel_loop, holes=[ellipse_loop])

model.synchronize()

model.add_physical([plane_surface], "Volume")
model.add_physical([channel_lines[0]], "i")
model.add_physical([channel_lines[2]], "o")
model.add_physical([channel_lines[3]], "t")
model.add_physical([channel_lines[1]], "b")
model.add_physical(ellipse_loop.curves, "c")

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
