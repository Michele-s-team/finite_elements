'''
This code generates a  ring  mesh with radial symmetry: symmetry is obtained by replicating a ring slice
The inner ring is tagged with tag 'circle_r_id', the outer ring is tagged with tag 'circle_R_id', and all radial lines (spokes) are tagged with 'radial_lines_id'

run with
python3 generate_ring_mesh.py [mesh resolution] [path where to read the ring slice] [path where to store the mesh]

where [path where to read the ring slice] is the output path of generate_mesh_ring_slice.py

Example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_ring_mesh.py ~/shared/generate_mesh/2d/ring/ring_slice/solution $SOLUTION_PATH
'''

import meshio
from fenics import *
import argparse
import math
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
parser.add_argument("input_dir")
parser.add_argument("output_dir")
args = parser.parse_args()

# mesh resolution
r = 1
R = 2
c_r = [0, 0, 0]
c_R = [0, 0, 0]
# the angle 2 \pi will be divided into N equal slices. Here N must be the same as in generate_mesh_ring_slice.py, and it must be a power of 2
N = 8
circle_r_id = 2
circle_R_id = 3
radial_lines_id = 4

M = int(np.round(math.log2(N)))
theta = 2 * np.pi / N

output_dir = args.output_dir
input_dir = args.input_dir
mesh_slice_file = input_dir + "/mesh.msh"
mesh_xdmf_file = output_dir + "/mesh.xdmf"

print(f'r = {r}, R = {R}, c_r = {c_r}, c_R = {c_R}, N = {N}, mesh_slice_file: {mesh_slice_file}')

# Load the mesh slice
mesh = meshio.read(mesh_slice_file)

# msh.print_mesh_info(mesh, 'Mesh before mirroring')

# initialize the loop over 0 <= theta < 2 pi by setting the initial values of the extremal points of the first ring slice
r_1 = np.array([r, 0])
r_2 = cal.R(theta).dot(r_1)
r_4 = np.array([R, 0])
r_3 = cal.R(theta).dot(r_4)

print('Looping through circle ...')

for i in range(1, M + 1):
    # at each step of this loop, a slice is doubled in size by mirroring, until a full ring is constructed

    # print(f'\t i = {i}')

    # set the extremal points of the new ring slice in terms of the old ones
    r_1 = np.copy(r_2)
    r_2 = cal.R(2 ** (i - 1) * theta).dot(r_1)
    r_4 = np.copy(r_3)
    r_3 = cal.R(2 ** (i - 1) * theta).dot(r_4)

    # define the axis of symmetry according to the current mirroring operation
    gamma_axis_of_symmetry = lambda t: cal.line(r_1, r_4, t)

    '''
    # define the function which tells whetehr a point lies on the current axis of symmetry
    def point_on_axis_of_symmetry(point):
        return cal.point_on_line(point, gamma_axis_of_symmetry)

    # define the function which makes current mirroring operation
    def mirror_function(point):
        return cal.mirror_point_line(point, gamma_axis_of_symmetry)


    # Mirror points across gamma_top
    old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
                                                                                                       mesh.point_data)
    msh.mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)
    msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)
    '''
    msh.mirror_mesh(mesh, gamma_axis_of_symmetry)

# tag circle_r: extract the lines whose starting point is part of  circle_r by considering its distance with respect to the circle center
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(np.linalg.norm(np.subtract(mesh.points[line[0]], c_r)), r) and np.isclose(np.linalg.norm(np.subtract(mesh.points[line[1]], c_r)), r)),
    circle_r_id, mesh
)

# tag circle_R: extract the lines whose starting point is part of  circle_R by considering its distance with respect to the circle center
msh.asssign_tag_to_lines(
    lambda line: (np.isclose(np.linalg.norm(np.subtract(mesh.points[line[0]], c_R)), R) and np.isclose(np.linalg.norm(np.subtract(mesh.points[line[1]], c_R)), R)),
    circle_R_id, mesh
)

# rag the radial lines
msh.asssign_tag_to_lines(lambda line: cal.line_is_radial(line, N, mesh), radial_lines_id, mesh)


print('... done.')
meshio.write(mesh_xdmf_file, mesh)  # XDMF for FEniCS

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
