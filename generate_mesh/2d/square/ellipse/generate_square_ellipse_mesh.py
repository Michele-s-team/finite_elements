'''
generate a mesh given by a square with a ellipse-shaped hole in it: the ellipse has the shape of an ellipse

run it with
    python3 generate_square_ellipse_mesh.py [resolution] [output directory]
example:
    SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_ellipse_mesh.py 0.1 $SOLUTION_PATH
'''

import meshio
import gmsh
import numpy as np
import pygmsh
import argparse

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)

# add '/' to output_directory if it is missing
output_directory = args.output_directory
output_directory = io.add_trailing_slash(output_directory)

mesh_file = output_directory + "mesh.msh"

# CHANGE PARAMETERS HERE
L = 1
h = 1
# ellipse center
c = [L / 2, h / 2, 0]
# ellipse semi-major axis
a = 0.2
# ellipse semi-minor axis
b = 0.1
# rotation angle of the ellipse with respect to the x axis: the ellipse will be rotated about its left focal point
phi = 0
# CHANGE PARAMETERS HERE


print("L = ", L)
print("h = ", h)
print(f"c = {c}, a = {a}, b = {b}, phi = {phi}")
print("resolution = ", resolution)
print(f'output_directory = "{output_directory}"')

# left focal point  of the ellipse
focus = np.subtract(c, [np.sqrt(a ** 2 - b ** 2), 0, 0])

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

my_points = [model.add_point((0, 0, 0), mesh_size=resolution),
             model.add_point((L, 0, 0), mesh_size=resolution),
             model.add_point((L, h, 0), mesh_size=resolution),
             model.add_point((0, h, 0), mesh_size=resolution)]

# Add lines between all points creating the rectangle
channel_lines = [model.add_line(my_points[i], my_points[i + 1])
                 for i in range(-1, len(my_points) - 1)]

channel_loop = model.add_curve_loop(channel_lines)

p_ellipse_c = model.add_point(
    np.add(focus, np.dot(cal.R_z(phi), np.subtract(c, focus)))
    , mesh_size=resolution)
p_ellipse_r = model.add_point(
    np.add(focus, np.dot(cal.R_z(phi), np.subtract(np.add(c, [a, 0, 0]), focus))),
    mesh_size=resolution)
p_ellipse_t = model.add_point(
    np.add(focus, np.dot(cal.R_z(phi), np.subtract(np.add(c, [0, b, 0]), focus))),
    mesh_size=resolution)
p_ellipse_l = model.add_point(
    np.add(focus, np.dot(cal.R_z(phi), np.subtract(np.subtract(c, [a, 0, 0]), focus))),
    mesh_size=resolution)
p_ellipse_b = model.add_point(
    np.add(focus, np.dot(cal.R_z(phi), np.subtract(np.subtract(c, [0, b, 0]), focus))),
    mesh_size=resolution)
# p_ellipse_focus = model.add_point(focus, mesh_size=resolution)

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

msh.write_mesh_to_csv(mesh_file, output_directory + 'line_vertices.csv')

gmsh.clear()
geometry.__exit__()

mesh_from_file = meshio.read(mesh_file)

line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_directory + "line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_directory + "triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_directory + "triangle_mesh.xdmf")
io.print_vertices_to_csv_file(mesh, output_directory + "vertices.csv")
