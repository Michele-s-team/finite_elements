'''
This code generates a square mesh with a circular hole in it,
which is symmetric with respect to both left <-> right and tob <-> bottom symmetries
Symmetry is enforced by mirroring the mesh points along two symmetry axes.

run with
    python3 generate_square_mesh.py [path where to read parameters] [output directory]

ATTENTION: [mesh resolution] must be small enough for the circle to be properly resolved
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_mesh.py $PARAMETERS_PATH $SOLUTION_PATH

This mesh can be checked with ~/shared/generate_mesh/2d/square/check_mesh_square.py


The quarter of a mesh will be saved in [path where to store the mesh] as quarter_mesh.msh. The complete mesh will be saved in
[path where to store the mesh] as mesh.xdmf, triangle_mesh.xdmf, line_mesh.xdmf and vertices.csv.
'''

import meshio
from fenics import *
import gmsh
import pygmsh
import sys
import numpy as np

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'

sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh as msh
import runtime_arguments_generate_mesh as rarg
import read_parameters_generate_mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')


# parser = argparse.ArgumentParser()
# parser.add_argument("resolution")
# parser.add_argument("output_dir")
# args = parser.parse_args()

# mesh resolution
# resolution = (float)(args.resolution)
# L = 1
# h = 1
# r = 0.25
x_coordinate_axis_of_symmetry = L / 2
y_coordinate_axis_of_symmetry = h / 2
c_r = [x_coordinate_axis_of_symmetry, y_coordinate_axis_of_symmetry, 0]

gamma_axis_of_symmetry_left_right = lambda t: cal.line([x_coordinate_axis_of_symmetry, 0], [x_coordinate_axis_of_symmetry, h], t)
gamma_axis_of_symmetry_top_bottom = lambda t: cal.line([0, y_coordinate_axis_of_symmetry], [rpam.parameters["L"], y_coordinate_axis_of_symmetry], t)

output_dir = io.add_trailing_slash(rarg.args.output_dir)
quarter_mesh_msh_file = output_dir + "quarter_mesh.msh"
mesh_xdmf_file = output_dir + "mesh.xdmf"

# print(f'rpam.parameters["L"] = {rpam.parameters["L"]}\nh = {h}\nr={r}\nc_r = {c_r}\nrpam.parameters["resolution"] = {rpam.parameters["resolution"]}\noutput directory = {output_dir}')

# The quarter mesh is generated used pygmsh and it is saved as quarter_mesh.msh

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

N = int(np.round(r * np.pi / 2 / rpam.parameters["resolution"]))

# construct a rectangle with vertices [rpam.parameters["L"],h/2], [rpam.parameters["L"],h], [rpam.parameters["L"]/2,h], [rpam.parameters["L"]/2,h/2]

quarter_rectangle_points = [model.add_point((rpam.parameters["L"], y_coordinate_axis_of_symmetry, 0), mesh_size=rpam.parameters["resolution"]),
                            model.add_point((rpam.parameters["L"], h, 0), mesh_size=rpam.parameters["resolution"]),
                            model.add_point((x_coordinate_axis_of_symmetry, h, 0), mesh_size=rpam.parameters["resolution"])
                            ]
model.synchronize()

quarter_circle_points = [
    model.add_point((c_r[0] + r * np.cos(np.pi / 2 * (N - i) / N), c_r[1] + r * np.sin(np.pi / 2 * (N - i) / N), 0), mesh_size=rpam.parameters["resolution"])
    for i in range(N + 1)]
model.synchronize()

quarter_rectangle_circle_points = quarter_rectangle_points + quarter_circle_points
quarter_rectangle_circle_lines = [model.add_line(quarter_rectangle_circle_points[i], quarter_rectangle_circle_points[i + 1])
                                  for i in range(-1, len(quarter_rectangle_circle_points) - 1)]

quarter_rectangle_circle_loop = model.add_curve_loop(quarter_rectangle_circle_lines)
quarter_rectangle_circle_surface = model.add_plane_surface(quarter_rectangle_circle_loop)

model.synchronize()

model.add_physical([quarter_rectangle_circle_surface], "Volume")
model.add_physical([quarter_rectangle_circle_lines[0]], "b")
model.add_physical([quarter_rectangle_circle_lines[1]], "r")
model.add_physical([quarter_rectangle_circle_lines[2]], "t")
model.add_physical(quarter_rectangle_circle_lines[3], "l")
model.add_physical(quarter_rectangle_circle_lines[4:], "quarter_circle")

geometry.generate_mesh(dim=2)
gmsh.write(quarter_mesh_msh_file)

# msh.print_mesh_lines_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

gmsh.clear()
geometry.__exit__()
'''
mesh_from_file = meshio.read(output_dir + '/quarter_mesh.msh')
line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_dir + "/triangle_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, output_dir + "/vertices.csv")
'''
###

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
interior_lines_id = 7

# Load the quarter mesh
mesh = meshio.read(quarter_mesh_msh_file)

# msh.print_mesh_info(mesh, 'Mesh before mirroring')


# mirror the quarter mesh ##

msh.mirror_mesh(mesh, gamma_axis_of_symmetry_left_right)
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

# tag circle
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(np.linalg.norm(np.subtract(mesh.points[line[0]], c_r)), r) and np.isclose(np.linalg.norm(np.subtract(mesh.points[line[1]], c_r)), r)),
    circle_id, mesh
)

# tag internal lines which result from mesh mirroring
# tag internal lines parallel to the y axis
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][0], x_coordinate_axis_of_symmetry, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], x_coordinate_axis_of_symmetry, rtol=cal.small_number))),
    interior_lines_id, mesh
)
# tag internal lines parallel to the x axis
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][1], y_coordinate_axis_of_symmetry, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], y_coordinate_axis_of_symmetry, rtol=cal.small_number))),
    interior_lines_id, mesh
)

meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS

print("Full mesh generated successfully!")

# msh.print_mesh_info(mesh, 'Mesh after mirroring')


# read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
mesh_from_file = meshio.read(mesh_xdmf_file)

line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_dir + "line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_dir + "triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_dir + "triangle_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, output_dir + "vertices.csv")
