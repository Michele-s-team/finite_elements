'''
Ths code generates a 2d mesh given by a square with a circular hole, where the mesh is enforced to  be symmetric
with respect to top <-> bottom by adding a set of auxiliary lines which run from the left to the right edge of the square

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/lines"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/lines/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH

The mesh generated with this code can be checked with ~/shared/generate_mesh/2d/square/check_mesh.py
'''

import meshio
import pygmsh
import gmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import list as lis
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

output_directory = io.add_trailing_slash(rarg.args.output_directory)

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


mesh_file = output_directory + "mesh.msh"

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# add a 0d object:

p_O = msh.add_point([0, 0, 0], gmsh.model.geo)

points_b, edge_b = msh.add_line_p_start_r_end(p_O, [rpam.parameters["L"], 0, 0], gmsh.model.geo)
points_r, segments_edge_r = msh.add_line_p_start_r_end_n(points_b[-1], [rpam.parameters["L"], rpam.parameters["h"], 0], rpam.parameters["n_lines_lr"], gmsh.model.geo)
points_t, edge_t = msh.add_line_p_start_r_end(points_r[-1], [0, rpam.parameters["h"], 0], gmsh.model.geo)
points_l, segments_edge_l = msh.add_line_p_start_p_end_n(points_t[-1], p_O, rpam.parameters["n_lines_lr"], gmsh.model.geo)

msh.print_point_list_info(points_l, 'points_l')
msh.print_point_list_info(points_r, 'points_r')

lines = lis.flatten_list([edge_b, segments_edge_r, edge_t, segments_edge_l])
print(f'lines = {lines}')

loop_square = gmsh.model.geo.add_curve_loop(lines)
gmsh.model.geo.synchronize()

points_circle, segments_circle = msh.add_circle_with_lines(rpam.parameters["c_r"], rpam.parameters["r"], rpam.parameters["n_lines_circle"], gmsh.model.geo)

circle_loop = gmsh.model.geo.add_curve_loop(segments_circle)
gmsh.model.geo.synchronize()

square_surface = gmsh.model.geo.add_plane_surface([loop_square, circle_loop])
gmsh.model.geo.synchronize()

# add auxiliary horizontal lines to make the mesh symmetric under top <-> bottom
lines_lr = []
for j in range(1, len(points_l) - 1):
    coord = msh.get_point_coordinates(points_l[j])
    if ((coord[1] < rpam.parameters["c_r"][1] - rpam.parameters["r"]) or (coord[1] > rpam.parameters["c_r"][1] + rpam.parameters["r"])):
        lines_lr.append((msh.add_line_p_start_p_end(points_l[j], points_r[len(points_l) - 1 - j], gmsh.model.geo))[1])

gmsh.model.mesh.embed(1, lines_lr, 2, square_surface)

print('Adding physical objects ...')
# add 0-dimensional objects
vertices = gmsh.model.getEntities(dim=0)
for i in range(len(vertices)):
    gmsh.model.addPhysicalGroup(vertices[i][0], [vertices[i][1]], i + 1)
    gmsh.model.setPhysicalName(vertices[i][0], i + 1, f"vertice_p_{i}")

# add 1-dimensional objects
lines = gmsh.model.getEntities(dim=1)

# tag the edges and the segments of the edges
msh.tag_group([edge_b], 1, 5, 'l_edge')
msh.tag_group(segments_edge_r, 1, 3, 'segments_r_edge')
msh.tag_group([edge_t], 1, 4, 't_edge')
msh.tag_group(segments_edge_l, 1, 2, 'segments_l_edge')

# tag the circle
msh.tag_group(segments_circle, 1, 6, 'segments_circle')

# add 2-dimensional objects
surfaces = gmsh.model.getEntities(dim=2)

gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], 1)
gmsh.model.setPhysicalName(surfaces[0][0], 1, "surface")

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

mesh_from_file = meshio.read(mesh_file)

msh.full_write(mesh_file, ['triangle', 'line', 'vertex'], metadata, output_directory, True)


msh.clear_gmsh(geometry)
