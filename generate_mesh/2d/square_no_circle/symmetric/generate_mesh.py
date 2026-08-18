'''
This code generates a  square mesh which is perfectly symmetric along both the x and y axis, i.e., it is a tiled repetition of the same
rectangular mesh unit
Symmetry is enforced by mirroring the mesh unit.
The surface is tagged with surface_id, the lines on the boundaries with l_edge_id, r_edge_id, t_edge_id and b_edge_id,
and all lines in the bulk of the mesh with internal_lines_id

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/symmetric"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/symmetric/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH

The half mesh will be saved in [path where to store the mesh] as half_mesh.msh. The complete mesh will be saved in
[path where to store the mesh] as mesh.xdmf, triangle_mesh.xdmf, line_mesh.xdmf and vertices.csv.
'''

from fenics import *
import gmsh
import meshio
import numpy as np
import os
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')


# number of tiles in which the mesh will be divided, along each axis
N = np.ceil(max(rpam.parameters["L"], rpam.parameters["h"]) / rpam.parameters["resolution"])
# given that  I will be mirroring (doubling) the mesh multiple times, N needs to be a power of two
log2_N = int(np.ceil(np.log2(N)))
N = 2 ** log2_N

L_unit = rpam.parameters["L"] / N
h_unit = rpam.parameters["h"] / N

output_dir = io.add_trailing_slash(rarg.args.output_directory)
unit_mesh_dir = io.add_trailing_slash(output_dir + 'unit_mesh')
os.makedirs(unit_mesh_dir, exist_ok=True)

unit_mesh_msh_file = unit_mesh_dir + "unit_mesh.msh"
mesh_xdmf_file = output_dir + "mesh.xdmf"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


# Unit mesh is generated used pygmsh and it's saved as unit_mesh.msh

geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# construct a rectangle with vertices [0,0], [L_unit_cell, 0], [L_unit_cell, h_unit_cell], [0, h_unit_cell]
unit_points = [model.add_point((0, 0, 0), mesh_size=rpam.parameters["resolution"]),
               model.add_point((L_unit, 0, 0), mesh_size=rpam.parameters["resolution"]),
               model.add_point((L_unit, h_unit, 0), mesh_size=rpam.parameters["resolution"]),
               model.add_point((0, h_unit, 0), mesh_size=rpam.parameters["resolution"]),
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

msh.print_mesh_edges_to_csv(unit_mesh_msh_file, unit_mesh_dir + 'edges.csv')

msh.clear_gmsh(geometry)

surface_id = 1
l_edge_id = 2
r_edge_id = 3
t_edge_id = 4
b_edge_id = 5
internal_lines_id = 6

# Load the unit mesh

mesh = meshio.read(unit_mesh_msh_file)

# msh.print_mesh_info(mesh, 'Mesh before mirroring')

## mirror the mesh ##
# mirror along y axis
for i in range(log2_N):
    gamma_axis_of_symmetry = lambda t: cal.line([0, (2 ** i) * h_unit], [L_unit, (2 ** i) * h_unit], t)
    msh.mirror_mesh(mesh, gamma_axis_of_symmetry)

# mirror along x axis
for i in range(log2_N):
    gamma_axis_of_symmetry = lambda t: cal.line([(2 ** i) * L_unit, 0], [(2 ** i) * L_unit, rpam.parameters["h"]], t)
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

# tag internal lines
msh.asssign_tag_to_lines(
    lambda line: (not ((np.isclose(mesh.points[line[0]][0], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], 0, rtol=cal.small_number))))) \
                 and (not ((np.isclose(mesh.points[line[0]][0], rpam.parameters["L"], rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][0], rpam.parameters["L"], rtol=cal.small_number))))) \
                 and (not ((np.isclose(mesh.points[line[0]][1], rpam.parameters["h"], rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], rpam.parameters["h"], rtol=cal.small_number))))) \
                 and (not ((np.isclose(mesh.points[line[0]][1], 0, rtol=cal.small_number) and (np.isclose(mesh.points[line[1]][1], 0, rtol=cal.small_number))))),
    internal_lines_id, mesh
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
