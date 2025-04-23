'''
This code generates a  ring0slice mesh

the tags are:
- surface: tag = 1
- inner ring arc : tag = 2
- outer ring arc : tag = 3
- top radial line : tag = 4
- bottom radial line : tag = 5

run with
python3 generate_mesh_ring_slice.py [mesh resolution] [path where to store the mesh]
Example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_ring_slice.py 0.3 $SOLUTION_PATH

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
r = 1
R = 2
c_r = [0, 0]
c_R = [0, 0]
N = 16
# N = int(np.round(r * np.pi / resolution))


output_dir = args.output_dir
mesh_file = output_dir + "/mesh.msh"
# mesh_xdmf_file = output_dir + "/mesh.xdmf"

surface_id = 1
circle_r_id = 2
circle_R_id = 3
line_t_id = 4
line_b_id = 5
ids = [1, line_b_id, circle_R_id, circle_r_id, line_t_id]

#  mesh is generated used pygmsh and it's saved in slice_mesh_msh_file
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()


print(f'r = {r}\nr = {R}\nc_r = {c_r}\nc_R = {c_R}\nresolution = {resolution}\noutput directory = {output_dir}')

theta = 2 * np.pi / N
phi = theta / 2
epsilon = 1e-2
p_c_r = model.add_point((c_r[0], c_r[1], 0))
p_c_R = model.add_point((c_R[0], c_R[1], 0))

# initialize the loop over 0 <= theta < 2 pi
r_1 = np.array([r, 0])
r_2 = cal.R(theta).dot(r_1)
r_4 = np.array([R, 0])
r_3 = cal.R(theta).dot(r_4)

# p_1 = gmsh.model.occ.addPoint( r_1[0], r_1[1], 0 )

p_1 = model.add_point((r_1[0], r_1[1], 0), mesh_size=resolution)
p_2 = model.add_point((r_2[0], r_2[1], 0), mesh_size=resolution)
p_3 = model.add_point((r_3[0], r_3[1], 0), mesh_size=resolution)
p_4 = model.add_point((r_4[0], r_4[1], 0), mesh_size=resolution)
model.synchronize()

arc_12 = model.add_circle_arc(p_1, p_c_r, p_2)
model.synchronize()

line_23 = model.add_line(p_2, p_3)
model.synchronize()

arc_34 = model.add_circle_arc(p_3, p_c_r, p_4)
model.synchronize()

line_41 = model.add_line(p_4, p_1)
model.synchronize()

slice_lines = [arc_12, line_23, arc_34, line_41]
slice_loop = model.add_curve_loop(slice_lines)
model.synchronize()

slice_surface = model.add_plane_surface(slice_loop)
model.synchronize()

model.add_physical([slice_surface], "Volume")
model.add_physical([slice_lines[0]], "r")
model.add_physical([slice_lines[2]], "R")
model.add_physical([slice_lines[1]], "top")
model.add_physical(slice_lines[3], "bottom")

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

gmsh.clear()
geometry.__exit__()

# Load the half-mesh
mesh = meshio.read(mesh_file)


line_mesh = msh.create_mesh(mesh, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_dir + "/triangle_mesh.xdmf")
io.print_vertices_to_csv_file(mesh, output_dir + "/vertices.csv")

# ################################################## mirror the mesh ##################################################

# # Mirror points across gamma_top
# old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
#                                                                                                    mesh.point_data)
# msh.mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)
# msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)
#
# # msh.asssign_tag_to_lines(
# #     lambda p_start, p_end: (np.isclose(p_start[1], 0, rtol=cal.small_number) and np.isclose(p_end[1], 0, rtol=1e-3)),
# #     b_edge_id, mesh
# # )
#
# meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS
# #
# print("Full mesh generated successfully!")
#
# '''
# print('********** Mesh after mirroring: **********')
# msh.print_mesh_element_types(mesh)
# msh.print_mesh_triangles(mesh)
# msh.print_mesh_vertices(mesh)
# '''
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
