'''
Ths code generates a 2d mesh given by a square with a circular hole, where the mesh is enforced to  be symmetric
with respect to  the center of the hole  by adding a set of concentric circles centerd at the hole's ceneter

run with
clear; clear; python3 generate_mesh.py [resolution] [number of segments for circles] [number of circles] [output directory]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py 0.1 32 4 $SOLUTION_PATH
'''

import gmsh
import pygmsh
import argparse
import numpy as np
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import list as lis
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("n_lines_circle")
parser.add_argument("n_aux_circles")
parser.add_argument("output_directory")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)
n_lines_circle = int(args.n_lines_circle)
n_aux_circles = int(args.n_aux_circles)
mesh_file = args.output_directory + "/mesh.msh"

# mesh parameters
# CHANGE PARAMETERS HERE
L = 0.5
h = 0.5
c_r = [L / 2, h / 2, 0]
r = 0.05
r_min = r
r_max = L/2 * 0.9
# CHANGE PARAMETERS HERE

print("L = ", L)
print("h = ", h)
print("r = ", r)
print("n_lines_circle = ", n_lines_circle)
print("n_aux_circles = ", n_aux_circles)

surface_id = 1

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# add a 0d object:

p_O = msh.add_point([0, 0, 0], gmsh.model.geo)

points_b, edge_b = msh.add_line_p_start_r_end(p_O, [L, 0, 0], gmsh.model.geo)
points_r, segments_edge_r = msh.add_line_p_start_r_end_n(points_b[-1], [L, h, 0], n_aux_circles, gmsh.model.geo)
points_t, edge_t = msh.add_line_p_start_r_end(points_r[-1], [0, h, 0], gmsh.model.geo)
points_l, segments_edge_l = msh.add_line_p_start_p_end_n(points_t[-1], p_O, n_aux_circles, gmsh.model.geo)

msh.print_point_list_info(points_l, 'points_l')
msh.print_point_list_info(points_r, 'points_r')

lines = lis.flatten_list([edge_b, segments_edge_r, edge_t, segments_edge_l])
print(f'lines = {lines}')

loop_square = gmsh.model.geo.add_curve_loop(lines)
gmsh.model.geo.synchronize()

points_circle, segments_circle = msh.add_circle_with_lines(c_r, r, n_lines_circle, gmsh.model.geo)

circle_loop = gmsh.model.geo.add_curve_loop(segments_circle)
gmsh.model.geo.synchronize()

# circle_lines, circle_loop = msh.add_circle_with_arcs(c_r, r, gmsh.model.geo)


square_surface = gmsh.model.geo.add_plane_surface([loop_square, circle_loop])
# square_surface = gmsh.model.geo.add_plane_surface([loop_square])
gmsh.model.geo.synchronize()

# gmsh.model.mesh.embed(1, [arc_rt, arc_tl, arc_lb, arc_br], 2, square_surface)

# line_aux = (msh.add_line_r_start_r_end([0.1, 0.1, 0], [0.9, 0.1, 0], gmsh.model.geo))[1]
# gmsh.model.mesh.embed(1, [line_aux], 2, square_surface)


# add auxiliary horizontal lines to make the mesh symmetric under top <-> bottom
aux_circles = []
for j in range(n_aux_circles):
    aux_r = r_min + (r_max - r_min) * j / (n_aux_circles - 1)
    if(aux_r > r and aux_r < L):
        print(f'aux_r = {aux_r}')
        aux_circles.append((msh.add_circle_with_lines(c_r, aux_r, n_lines_circle, gmsh.model.geo))[1])

gmsh.model.mesh.embed(1, lis.flatten_list(aux_circles), 2, square_surface)
gmsh.model.geo.synchronize()



print('Adding physical objects ...')
# add 0-dimensional objects
vertices = gmsh.model.getEntities(dim=0)
for i in range(len(vertices)):
    gmsh.model.addPhysicalGroup(vertices[i][0], [vertices[i][1]], i + 1)
    gmsh.model.setPhysicalName(vertices[i][0], i + 1, f"vertice_p_{i}")

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# add each of the four edges of the square
# for i in range(4):
#     msh.print_line_info(lines[i][1], f'linea_{i}')
#
#     gmsh.model.addPhysicalGroup(lines[i][0], [lines[i][1]], i + 1)
#     gmsh.model.setPhysicalName(lines[i][0], i + 1, f"line_{i + 1}")

# tag the edges and the segments of the edges
msh.tag_group(segments_edge_l, 1, 2, 'segments_l_edge')
msh.tag_group(segments_edge_r, 1, 3, 'segments_r_edge')
msh.tag_group([edge_t], 1, 4, 't_edge')
msh.tag_group([edge_b], 1, 5, 'b_edge')
msh.tag_group(segments_circle, 1, 6, 'segments_circle')

# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], surface_id)
gmsh.model.setPhysicalName(surfaces[0][0], surface_id, "superficie")

print('... done.')

# set the resolution
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", [square_surface])

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution / 2)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", h)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", L)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

# write mesh components to file
msh.write_mesh_components(mesh_file, "solution/triangle_mesh.xdmf", "triangle", True)
msh.write_mesh_components(mesh_file, "solution/line_mesh.xdmf", "line", True)
msh.write_mesh_components(mesh_file, "solution/vertex_mesh.xdmf", "vertex", True)

msh.write_mesh_to_csv(mesh_file, 'solution/line_vertices.csv')

model.__exit__()
