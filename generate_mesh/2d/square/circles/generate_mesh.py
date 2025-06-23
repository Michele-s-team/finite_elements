'''
Ths code generates a 2d mesh given by a square with a circular hole, where the mesh is enforced to  be symmetric
with respect to  the center of the hole  by adding a set of concentric circles centered at the hole's ceneter

Run with
    clear; clear; python3 generate_mesh.py [resolution] [number of segments for circles] [number of circles] [output directory]
Example:
    clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py 0.1 32 4 $SOLUTION_PATH
'''

import gmsh
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import list as lis
import mesh as msh
import runtime_arguments_generate_mesh as rarg
import read_parameters_generate_mesh as rpam

# print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')
# parser = argparse.ArgumentParser()
# parser.add_argument("resolution")
# parser.add_argument("n_lines_circle")
# parser.add_argument("n_aux_circles")
# parser.add_argument("output_directory")
# args = parser.parse_args()

output_directory = io.add_trailing_slash(rarg.args.output_directory)

# mesh resolution
# resolution = (float)(rarg.args.resolution)
# n_lines_circle = int(rarg.args.n_lines_circle)
# n_aux_circles = int(rarg.args.n_aux_circles)
mesh_file = output_directory + "mesh.msh"

# mesh parameters
# CHANGE PARAMETERS HERE
# L = 1
# h = 1
# c_r = [L / 2, h / 2, 0]
# r = 0.25
# r_min = r
# r_max = L / 2 * 0.9
# CHANGE PARAMETERS HERE

# print("L = ", L)
# print("h = ", h)
# print("r = ", r)
# print("n_lines_circle = ", n_lines_circle)
# print("n_aux_circles = ", n_aux_circles)

surface_id = 1

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# add a 0d object:

p_O = msh.add_point([0, 0, 0], gmsh.model.geo)

points_b, edge_b = msh.add_line_p_start_r_end(p_O, [rpam.parameters["L"], 0, 0], gmsh.model.geo)
points_r, segments_edge_r = msh.add_line_p_start_r_end_n(points_b[-1], [rpam.parameters["L"], rpam.parameters["h"], 0], rpam.parameters["n_aux_circles"], gmsh.model.geo)
points_t, edge_t = msh.add_line_p_start_r_end(points_r[-1], [0, rpam.parameters["h"], 0], gmsh.model.geo)
points_l, segments_edge_l = msh.add_line_p_start_p_end_n(points_t[-1], p_O, rpam.parameters["n_aux_circles"], gmsh.model.geo)

msh.print_point_list_info(points_l, 'points_l')
msh.print_point_list_info(points_r, 'points_r')

lines = lis.flatten_list([edge_b, segments_edge_r, edge_t, segments_edge_l])
print(f'lines = {lines}')

loop_square = gmsh.model.geo.add_curve_loop(lines)
gmsh.model.geo.synchronize()

points_circle, segments_circle = msh.add_circle_with_lines(rpam.parameters["c_r"], rpam.parameters["r"], rpam.parameters["n_lines_circle"], gmsh.model.geo)

circle_loop = gmsh.model.geo.add_curve_loop(segments_circle)
gmsh.model.geo.synchronize()

# circle_lines, circle_loop = msh.add_circle_with_arcs(rpam.parameters["c_r"], rpam.parameters["r"], gmsh.model.geo)


square_surface = gmsh.model.geo.add_plane_surface([loop_square, circle_loop])
# square_surface = gmsh.model.geo.add_plane_surface([loop_square])
gmsh.model.geo.synchronize()

# gmsh.model.mesh.embed(1, [arc_rt, arc_tl, arc_lb, arc_br], 2, square_surface)

# line_aux = (msh.add_line_r_start_r_end([0.1, 0.1, 0], [0.9, 0.1, 0], gmsh.model.geo))[1]
# gmsh.model.mesh.embed(1, [line_aux], 2, square_surface)


# add auxiliary horizontal lines to make the mesh symmetric under top <-> bottom
aux_circles = []
for j in range(rpam.parameters["n_aux_circles"]):
    aux_r = r_min + (r_max - r_min) * j / (rpam.parameters["n_aux_circles"] - 1)
    if (aux_r > rpam.parameters["r"] and aux_r < rpam.parameters["L"]):
        print(f'aux_r = {aux_r}')
        aux_circles.append((msh.add_circle_with_lines(rpam.parameters["c_r"], aux_r, rpam.parameters["n_lines_circle"], gmsh.model.geo))[1])

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
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution"] / 2)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", rpam.parameters["h"])
gmsh.model.mesh.field.setNumber(threshold, "DistMax", rpam.parameters["L"])

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.geo.synchronize()

geometry.generate_mesh(dim=2)
gmsh.write(mesh_file)

# write mesh components to file
# msh.write_mesh_components(mesh_file, output_directory + "triangle_mesh.xdmf", "triangle", True)
# msh.write_mesh_components(mesh_file, output_directory + "line_mesh.xdmf", "line", True)
# msh.write_mesh_components(mesh_file, output_directory + "vertex_mesh.xdmf", "vertex", True)
#
# msh.print_mesh_lines_to_csv(mesh_file, output_directory + 'line_vertices.csv')
#
#
# # print the mesh vertices to file
# mesh = msh.read_mesh(output_directory + "triangle_mesh.xdmf")
# io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")

msh.full_write(mesh_file, ['triangle', 'line', 'vertex'], rpam.parameters, output_directory, True)


model.__exit__()
