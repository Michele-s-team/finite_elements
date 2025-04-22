'''
This code generates a  ring mesh with radial symmetry
Symmetry is enforced by mirroring the mesh points across multiple slices

run with
python3 generate_ring_mesh.py [mesh resolution] [path where to store the mesh]
Example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_ring_mesh.py 0.3 $SOLUTION_PATH

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
c_r = [0,0]
c_R = [0,0]


output_dir = args.output_dir
slice_mesh_msh_file = output_dir + "/slice_mesh.msh"
mesh_xdmf_file = output_dir + "/mesh.xdmf"

print(f'r = {r}\nr = {R}\nc_r = {c_r}\nc_R = {c_R}\nresolution = {resolution}\noutput directory = {output_dir}')

'''
slice mesh is generated used pygmsh and it's saved in slice_mesh_msh_file
'''
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

N = int(np.round(r * np.pi / resolution))




theta = 2 * np.pi / N
phi = theta / 2
epsilon = 1e-2
p_c_r = model.add_point(( c_r[0], c_r[1], 0 ))
p_c_R = model.add_point(( c_R[0], c_R[1], 0 ))


# def Q(theta):
#     return np.array( [[np.cos( theta ), -np.sin( theta )], [np.sin( theta ), np.cos( theta )]] )


# initialize the loop over 0 <= theta < 2 pi
r_1 = np.array( [r, 0] )
r_2 = cal.R( theta ).dot( r_1 )
r_4 = np.array( [R, 0] )
r_3 = cal.R( theta ).dot( r_4 )

# p_1 = gmsh.model.occ.addPoint( r_1[0], r_1[1], 0 )

p_1 = model.add_point((r_1[0], r_1[1], 0))
p_2 = model.add_point((r_2[0], r_2[1], 0))
p_3 = model.add_point((r_3[0], r_3[1], 0))
p_4 = model.add_point((r_4[0], r_4[1], 0))
model.synchronize()

arc_12 = model.add_circle_arc( p_1, p_c_r, p_2 )
model.synchronize()

line_23 = model.add_line( p_2, p_3 )
model.synchronize()

arc_34 = model.add_circle_arc( p_3, p_c_r, p_4 )
model.synchronize()

line_41 = model.add_line( p_4, p_1 )
model.synchronize()

slice_lines =  [arc_12, line_23, arc_34, line_41]
slice_loop = model.add_curve_loop(slice_lines)
model.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(slice_mesh_msh_file)


#
# surfaces = []
#
# # loop through N-1 slices of the ring
# for i in range( N - 1 ):
#     print( f"Adding slice #{i} ... " )
#
#     print( f"\tr_1 = {r_1}" )
#     print( f"\tr_2 = {r_2}" )
#     print( f"\tr_3 = {r_3}" )
#     print( f"\tr_4 = {r_4}" )
#
#     arc_12 = gmsh.model.occ.addCircleArc( p_1, p_c_r, p_2 )
#     line_23 = gmsh.model.occ.addLine( p_2, p_3 )
#     arc_34 = gmsh.model.occ.addCircleArc( p_3, p_c_R, p_4 )
#     line_41 = gmsh.model.occ.addLine( p_4, p_1 )
#     gmsh.model.occ.synchronize()
#
#     loop = gmsh.model.occ.addCurveLoop( [arc_12, line_23, arc_34, line_41] )
#     surfaces.append( gmsh.model.occ.addPlaneSurface( [loop] ) )
#     gmsh.model.occ.synchronize()
#
#     r_2 = Q( theta ).dot( r_2 )
#     r_3 = Q( theta ).dot( r_3 )
#
#     p_1 = p_2
#     p_2 = gmsh.model.occ.addPoint( r_2[0], r_2[1], 0 )
#     p_4 = p_3
#     p_3 = gmsh.model.occ.addPoint( r_3[0], r_3[1], 0 )
#     gmsh.model.occ.synchronize()
#
#     print( "...done" )
#
# # close the loop with a special curve addition for the last slice
# arc_12 = gmsh.model.occ.addCircleArc( p_1, p_c_r, p_1_start )
# line_23 = gmsh.model.occ.addLine( p_1_start, p_4_start )
# arc_34 = gmsh.model.occ.addCircleArc( p_4_start, p_c_R, p_4 )
# line_41 = gmsh.model.occ.addLine( p_4, p_1 )
# gmsh.model.occ.synchronize()
#
# loop = gmsh.model.occ.addCurveLoop( [arc_12, line_23, arc_34, line_41] )
# surfaces.append( gmsh.model.occ.addPlaneSurface( [loop] ) )
# gmsh.model.occ.synchronize()


#
# # construct a rectangle with vertices [L,h/2], [L,h], [0,h], [0,h/2]
#
# half_rectangle_points = [model.add_point((L, y_coordinate_axis_of_symmetry, 0), mesh_size=resolution),
#                          model.add_point((L, h, 0), mesh_size=resolution),
#                          model.add_point((0, h, 0), mesh_size=resolution),
#                          model.add_point((0, y_coordinate_axis_of_symmetry, 0), mesh_size=resolution),
#                          ]
# model.synchronize()
#
# half_circle_points = [
#     model.add_point((c_r[0] + -r * np.cos(np.pi * i / N), c_r[1] + r * np.sin(np.pi * i / N), 0), mesh_size=resolution)
#     for i in range(N + 1)]
# model.synchronize()
#
# half_rectangle_circle_points = half_rectangle_points + half_circle_points
# half_rectangle_circle_lines = [model.add_line(half_rectangle_circle_points[i], half_rectangle_circle_points[i + 1])
#                                for i in range(-1, len(half_rectangle_circle_points) - 1)]
#
# half_rectangle_circle_loop = model.add_curve_loop(half_rectangle_circle_lines)
# half_rectangle_circle_surface = model.add_plane_surface(half_rectangle_circle_loop)
#
# model.synchronize()
#
# model.add_physical([half_rectangle_circle_surface], "Volume")
# model.add_physical([half_rectangle_circle_lines[1]], "r")
# model.add_physical([half_rectangle_circle_lines[3]], "l")
# model.add_physical([half_rectangle_circle_lines[2]], "t")
# # model.add_physical( [channel_lines[4],channel_lines[0]], "b" )
# model.add_physical(half_rectangle_circle_lines[5:], "c")
#
# geometry.generate_mesh(dim=2)
# gmsh.write(sliced()_mesh_msh_file)
#
# # msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )
#
# gmsh.clear()
# geometry.__exit__()
#
# '''
# duplicate the points and cells with the respective tags and ids
# The new mesh inherits the ids (physical id used for measure definiton) of the original one,
# except for the new physical objects that are generated from reflection (e.g. the b line)
#
# In particular the rule 4:5 implies that the lines that in the original mesh where
# in the physical group 4 (top lines), when reflected, they will be assigned the id 5 (used to define measure in the bottom line)
#
# Here the lines are tagged as follows:
# - volume: id = 1
# - b edge: id = 4: now it is set to np.nan is because the l edge generated here, in the half mesh, will be immaterial when the mesh will be mirrored ->
#   a proper ID will be assigned to it later
# - r edge: id = 2
# - t edge: id = 3
# - l edge: id = 1
# - circle: id = 5
# '''
# surface_id = 1
# l_edge_id = 2
# r_edge_id = 3
# t_edge_id = 4
# b_edge_id = 5
# circle_id = 6
# ids = [1, np.nan, r_edge_id, l_edge_id, t_edge_id, circle_id]
# # Load the half-mesh
# mesh = meshio.read(sliced()_mesh_msh_file)
#
# '''
# print('********** Mesh before mirroring: **********')
# msh.print_mesh_element_types(mesh)
# msh.print_mesh_triangles(mesh)
# msh.print_mesh_vertices(mesh)
# '''
#
# ################################################## mirror the mesh ##################################################
#
# # Mirror points across X=0
# old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(y_coordinate_axis_of_symmetry, h, mesh.points,
#                                                                                                    mesh.point_data)
#
# old_triangles = mesh.cells_dict['triangle']
# old_lines = mesh.cells_dict['line']
#
# # duplicate cell blocks of type 'triangle'
# new_triangles = np.copy(old_triangles)
# # run through the old triangles
# for i in range(np.shape(new_triangles)[0]):
#     # for each old triangle, run through each of its three vertices
#     for j in range(3):
#         '''
#         assign to the new triangle the vertex tag of the old triangle, mapped towards the vertex tags of the mirrored vertices
#         In this way, one reconstructs the same pattern as the old triangles, for the flipped part of the mesh
#         '''
#         new_triangles[i, j] = non_mirrored_plus_new_points_indices[old_triangles[i, j]]
#
# mesh.points = old_plus_new_points
# mesh.point_data['gmsh:dim_tags'] = np.vstack((mesh.point_data['gmsh:dim_tags'], mirrored_point_data))
# mesh.cells[-1] = meshio.CellBlock("triangle", np.vstack((old_triangles, new_triangles)))
# N = np.shape(mesh.cells[-1].data)[0]
# mesh.cell_data['gmsh:physical'][-1] = np.array([mesh.cell_data['gmsh:physical'][-1][0]] * N)
# mesh.cell_data['gmsh:geometrical'][-1] = np.array([mesh.cell_data['gmsh:geometrical'][-1][0]] * N)
#
# # duplicate cell blocks of type 'line'
# for j in range(len(mesh.cells)):
#     if mesh.cells[j].type == 'line':
#         lines = np.copy(mesh.cells[j].data)
#         filtered_lines = []
#         for i in range(np.shape(lines)[0]):
#             f = [mesh.points[lines[i, k]][1] != 0 for k in range(2)]
#             if f[0] or f[1]:
#                 filtered_lines.append([non_mirrored_plus_new_points_indices[lines[i, 0]],
#                                        non_mirrored_plus_new_points_indices[lines[i, 1]]])
#         filtered_lines = np.array(filtered_lines)
#         mesh.cells[j] = meshio.CellBlock("line", np.vstack((lines, filtered_lines)))
#         N = np.shape(mesh.cells[j].data)[0]
#         mesh.cell_data['gmsh:physical'][j] = np.array([ids[mesh.cell_data['gmsh:physical'][j][0]]] * N)
#         mesh.cell_data['gmsh:geometrical'][j] = np.array([mesh.cell_data['gmsh:geometrical'][j][0]] * N)
#
# msh.asssign_tag_to_lines(
#     lambda p_start, p_end: (np.isclose(p_start[1], 0, rtol=cal.small_number) and np.isclose(p_end[1], 0, rtol=1e-3)),
#     b_edge_id, mesh
# )
#
# meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS
#
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
