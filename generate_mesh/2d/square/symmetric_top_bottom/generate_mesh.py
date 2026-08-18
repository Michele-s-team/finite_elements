'''
This code generates a  square mesh with a circular hole in it, which is symmetric with respect to top <-> bottom symmetry
Symmetry is enforced by mirroring the mesh points along a symetry axis.

ATTENTION:  in the parameters file 'resolution' must be small enough for the circle to be properly resolved
ATTENTION: c_r[1] in mesh_parameters must be equal to h/2, otherwise symmetry of the mesh would not make sense

Run with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_top_bottom"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_top_bottom/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH

The half mesh will be saved in [path where to store the mesh] as half_mesh.msh. The complete mesh will be saved in
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
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

if rpam.parameters["c_r"][1] !=  rpam.parameters["h"]/2:
    print("ERROR: c_r[1] is not equal to  h/2")

gamma_axis_of_symmetry = lambda t: cal.line([0, rpam.parameters["c_r"][1]], [rpam.parameters["L"], rpam.parameters["c_r"][1]], t)

output_dir = io.add_trailing_slash(rarg.args.output_directory)

half_mesh_msh_file = output_dir + "half_mesh.msh"
mesh_xdmf_file = output_dir + "mesh.xdmf"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'



# Half mesh is generated used pygmsh and it's saved as mesh.msh

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

N = int(np.round(rpam.parameters["r"] * np.pi / rpam.parameters["resolution"]))

# construct a rectangle with vertices [rpam.parameters["L"],h/2], [rpam.parameters["L"],h], [0,h], [0,h/2]

half_rectangle_points = [model.add_point((rpam.parameters["L"], rpam.parameters["c_r"][1], 0), mesh_size=rpam.parameters["resolution"]),
                         model.add_point((rpam.parameters["L"], rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
                         model.add_point((0, rpam.parameters["h"], 0), mesh_size=rpam.parameters["resolution"]),
                         model.add_point((0, rpam.parameters["c_r"][1], 0), mesh_size=rpam.parameters["resolution"]),
                         ]
model.synchronize()

half_circle_points = [
    model.add_point((rpam.parameters["c_r"][0] + -rpam.parameters["r"] * np.cos(np.pi * i / N), rpam.parameters["c_r"][1] + rpam.parameters["r"] * np.sin(np.pi * i / N), 0), mesh_size=rpam.parameters["resolution"])
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
gmsh.write(half_mesh_msh_file)

msh.print_mesh_edges_to_csv(half_mesh_msh_file, output_dir + 'edges.csv')

msh.clear_gmsh(geometry)

'''
duplicate the points and cells with the respective tags and ids
The new mesh inherits the ids (physical id used for measure definiton) of the original one,
except for the new physical objects that are generated from reflection (e.g. the b line)
'''
l_edge_id = 2
r_edge_id = 3
t_edge_id = 4
b_edge_id = 5
circle_id = 6

# Load the half mesh
mesh = meshio.read(half_mesh_msh_file)

# msh.print_mesh_info(mesh, 'Mesh before mirroring')


## mirror the mesh ##
'''
old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
                                                                                                   mesh.point_data)
msh.mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)
msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)
'''
msh.mirror_mesh(mesh, gamma_axis_of_symmetry)

# tag l edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][0], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], 0, rtol=cal.small_number))),
    l_edge_id, mesh
)

# tag r edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][0], rpam.parameters["L"], rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], rpam.parameters["L"], rtol=cal.small_number))),
    r_edge_id, mesh
)

# tag t edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][1], rpam.parameters["h"], rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], rpam.parameters["h"], rtol=cal.small_number))),
    t_edge_id, mesh
)

# tag b edge
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(mesh.points[line[0]][1], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], 0, rtol=cal.small_number))),
    b_edge_id, mesh
)

msh.asssign_tag_to_lines(
    lambda line: np.linalg.norm(np.subtract(mesh.points[line[0]], rpam.parameters["c_r"])) < (rpam.parameters["r"] + cal.min_dist_c_r_rectangle(rpam.parameters["L"], rpam.parameters["h"], rpam.parameters["c_r"])) / 2,
    circle_id, mesh
)

meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS

print("Full mesh generated successfully!")

# read the mesh.xdmf file and generate line_mesh.xdmf and triangle_mesh.xdmf
mesh_from_file = meshio.read(mesh_xdmf_file)

line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_dir + "line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_dir + "triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_dir + "triangle_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, output_dir + "vertices.csv")

# print mesh metadata
io.write_parameters_to_csv_file(output_dir + "mesh_metadata.csv", metadata)