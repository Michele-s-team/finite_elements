'''
This code generates a square mesh with a circular hole in it,
which is symmetric with respect to both left <-> right and tob <-> bottom symmetries
Symmetry is enforced by mirroring the mesh points along two symmetry axes.

run with
python3 generate_square_mesh.py [mesh resolution] [path where to store the mesh]
ATTENTION: [mesh resolution] must be small enough for the circle to be properly resolved
Example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_mesh.py 0.3 $SOLUTION_PATH

The quarter of a mesh will be saved in [path where to store the mesh] as quarter_mesh.msh. The complete mesh will be saved in
[path where to store the mesh] as mesh.xdmf, triangle_mesh.xdmf, line_mesh.xdmf and vertices.csv.
'''

import meshio
from fenics import *
import gmsh  # main tool
import pygmsh  # wrapper for gmsh
import argparse
import sys
import numpy as np

# add the path where to find the shared modules
# gaetano's path
# module_path = '/home/tanos/Thesis/finite_elements/modules/'
# michele's path
module_path = '/home/fenics/shared/modules'

sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_dir")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)
r = 0.25
L = 1
h = 1
x_coordinate_axis_of_symmetry = L / 2
y_coordinate_axis_of_symmetry = h / 2
c_r = [x_coordinate_axis_of_symmetry, y_coordinate_axis_of_symmetry, 0]

gamma_axis_of_symmetry_left_right = lambda t: cal.line([x_coordinate_axis_of_symmetry, 0], [x_coordinate_axis_of_symmetry, h], t)
gamma_axis_of_symmetry_top_bottom = lambda t: cal.line([0, y_coordinate_axis_of_symmetry], [L, y_coordinate_axis_of_symmetry], t)

output_dir = args.output_dir
quarter_mesh_msh_file = output_dir + "/quarter_mesh.msh"
mesh_xdmf_file = output_dir + "/mesh.xdmf"

print(f'L = {L}\nh = {h}\nc_r = {c_r}\nresolution = {resolution}\noutput directory = {output_dir}')

# The quarter mesh is generated used pygmsh and it is saved as quarter_mesh.msh

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

N = int(np.round(r * np.pi / resolution))

# construct a rectangle with vertices [L,h/2], [L,h], [0,h], [0,h/2]

half_rectangle_points = [model.add_point((L, y_coordinate_axis_of_symmetry, 0), mesh_size=resolution),
                         model.add_point((L, h, 0), mesh_size=resolution),
                         model.add_point((0, h, 0), mesh_size=resolution),
                         model.add_point((0, y_coordinate_axis_of_symmetry, 0), mesh_size=resolution),
                         ]
model.synchronize()

half_circle_points = [
    model.add_point((c_r[0] + -r * np.cos(np.pi * i / N), c_r[1] + r * np.sin(np.pi * i / N), 0), mesh_size=resolution)
    for i in range(N + 1)]
model.synchronize()

half_rectangle_circle_points = half_rectangle_points + half_circle_points
half_rectangle_circle_lines = [model.add_line(half_rectangle_circle_points[i], half_rectangle_circle_points[i + 1])
                               for i in range(-1, len(half_rectangle_circle_points) - 1)]

half_rectangle_circle_loop = model.add_curve_loop(half_rectangle_circle_lines)
half_rectangle_circle_surface = model.add_plane_surface(half_rectangle_circle_loop)

model.synchronize()

model.add_physical([half_rectangle_circle_surface], "Volume")
model.add_physical([half_rectangle_circle_lines[1]], "r")
model.add_physical([half_rectangle_circle_lines[3]], "l")
model.add_physical([half_rectangle_circle_lines[2]], "t")
model.add_physical(half_rectangle_circle_lines[5:], "c")

geometry.generate_mesh(dim=2)
gmsh.write(quarter_mesh_msh_file)

# msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

gmsh.clear()
geometry.__exit__()

'''
duplicate the points and cells with the respective tags and ids
The new mesh inherits the ids (physical id used for measure definiton) of the original one,
except for the new physical objects that are generated from reflection (e.g. the b line)
'''
surface_id = 1
l_edge_id = 2
r_edge_id = 3
t_edge_id = 4
b_edge_id = 5
circle_id = 6

# Load the half mesh
mesh = meshio.read(quarter_mesh_msh_file)

# msh.print_mesh_info(mesh, 'Mesh before mirroring')


## mirror the mesh ##
'''
old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
                                                                                                   mesh.point_data)
msh.mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)
msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)
'''
msh.mirror_mesh(mesh, gamma_axis_of_symmetry_top_bottom)

# tag l edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][0], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], 0, rtol=cal.small_number))),
    l_edge_id, mesh
)

# tag r edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][0], L, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], L, rtol=cal.small_number))),
    r_edge_id, mesh
)

# tag t edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][1], h, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], h, rtol=cal.small_number))),
    t_edge_id, mesh
)

# tag b edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][1], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], 0, rtol=cal.small_number))),
    b_edge_id, mesh
)


msh.asssign_tag_to_lines(
    lambda line: np.linalg.norm(np.subtract(mesh.points[line[0]], c_r)) < (r + cal.min_dist_c_r_rectangle(L, h, c_r)) / 2,
    circle_id, mesh
)

meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS

print("Full mesh generated successfully!")

# msh.print_mesh_info(mesh, 'Mesh after mirroring')


# read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
mesh_from_file = meshio.read(mesh_xdmf_file)

line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_dir + "/triangle_mesh.xdmf")
io.print_vertices_to_csv_file(mesh, output_dir + "/vertices.csv")
