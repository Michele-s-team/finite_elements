'''
This code generates a  square mesh which is perfectly symmetric along both the x and y axis, i.e., it is a tiled repetition of the same
rectangular mesh unit
Symmetry is enforced by mirroring the mesh unit

run with
python3 generate_square_no_circle_mesh.py [mesh resolution] [path where to store the mesh]

Example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_no_circle_mesh.py 0.3 $SOLUTION_PATH

The half mesh will be saved in [path where to store the mesh] as half_mesh.msh. The complete mesh will be saved in
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

L = 1
h = 1

L_unit_cell = resolution

if h < L:
    h_unit_cell = resolution * h / L
else:
    h_unit_cell = resolution * L / h

gamma_axis_of_symmetry = lambda t: cal.line([0, h_unit_cell], [L_unit_cell, h_unit_cell], t)

output_dir = args.output_dir
unit_mesh_msh_file = output_dir + "/unit_mesh.msh"
mesh_xdmf_file = output_dir + "/mesh.xdmf"

print(f'L = {L}\nh = {h}\nresolution = {resolution}\noutput directory = {output_dir}')

# Unit mesh is generated used pygmsh and it's saved as unit_mesh.msh

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# construct a rectangle with vertices [0,0], [L_unit_cell, 0], [L_unit_cell, h_unit_cell], [0, h_unit_cell]
unit_points = [model.add_point((0, 0, 0), mesh_size=resolution),
               model.add_point((L_unit_cell, 0, 0), mesh_size=resolution),
               model.add_point((L_unit_cell, h_unit_cell, 0), mesh_size=resolution),
               model.add_point((0, h_unit_cell, 0), mesh_size=resolution),
               ]
model.synchronize()

unit_lines = [model.add_line(unit_points[i], unit_points[i + 1])
              for i in range(-1, len(unit_points) - 1)]

unit_loop = model.add_curve_loop(unit_lines)
unit_surface = model.add_plane_surface(unit_loop)

model.synchronize()

model.add_physical([unit_surface], 'Volume')
model.add_physical([unit_lines[0]], 'l')
model.add_physical([unit_lines[2]], 'r')
model.add_physical([unit_lines[3]], 't')
model.add_physical([unit_lines[1]], 'b')

geometry.generate_mesh(dim=2)
gmsh.write(unit_mesh_msh_file)

gmsh.clear()
geometry.__exit__()


# # read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
mesh = meshio.read(unit_mesh_msh_file)

line_mesh = msh.create_mesh(mesh, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)



######
#
# '''
# duplicate the points and cells with the respective tags and ids
# The new mesh inherits the ids (physical id used for measure definiton) of the original one,
# except for the new physical objects that are generated from reflection (e.g. the b line)
# '''
# surface_id = 1
# l_edge_id = 2
# r_edge_id = 3
# t_edge_id = 4
# b_edge_id = 5
# circle_id = 6
#
# # Load the half mesh
# mesh = meshio.read(unit_mesh_msh_file)
#
# # msh.print_mesh_info(mesh, 'Mesh before mirroring')
#
#
# ## mirror the mesh ##
# '''
# old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
#                                                                                                    mesh.point_data)
# msh.mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)
# msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)
# '''
# msh.mirror_mesh(mesh, gamma_axis_of_symmetry)
#
# # tag l edge
# msh.asssign_tag_to_lines(
#     lambda line: (np.isclose(mesh.points[line[0]][0], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], 0, rtol=cal.small_number))),
#     l_edge_id, mesh
# )
#
# # tag r edge
# msh.asssign_tag_to_lines(
#     lambda line: (np.isclose(mesh.points[line[0]][0], L, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], L, rtol=cal.small_number))),
#     r_edge_id, mesh
# )
#
# # tag t edge
# msh.asssign_tag_to_lines(
#     lambda line: (np.isclose(mesh.points[line[0]][1], h, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], h, rtol=cal.small_number))),
#     t_edge_id, mesh
# )
#
# # tag b edge
# msh.asssign_tag_to_lines(
#     lambda line: (np.isclose(mesh.points[line[0]][1], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], 0, rtol=cal.small_number))),
#     b_edge_id, mesh
# )
#
# msh.asssign_tag_to_lines(
#     lambda line: np.linalg.norm(np.subtract(mesh.points[line[0]], c_r)) < (r + cal.min_dist_c_r_rectangle(L, h, c_r)) / 2,
#     circle_id, mesh
# )
#
# meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS
#
# print("Full mesh generated successfully!")
#
# # msh.print_mesh_info(mesh, 'Mesh after mirroring')
#
#
# # read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
# mesh_from_file = meshio.read(mesh_xdmf_file)
#
# line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
# meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)
#
# triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
# meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)
#
# # print the mesh vertices to file
# mesh = msh.read_mesh(output_dir + "/triangle_mesh.xdmf")
# io.print_vertices_to_csv_file(mesh, output_dir + "/vertices.csv")
