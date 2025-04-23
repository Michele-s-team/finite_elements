'''
This code generates a  ring  mesh with radial symmetry: symmetry is obtained by replicating a ring slice


run with
python3 generate_mesh_ring.py [mesh resolution] [path where to read the ring slice] [path where to store the mesh]
Example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_ring.py 0.1 ~/shared/generate-mesh/2d/ring_slice/solution $SOLUTION_PATH

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
parser.add_argument("input_dir")
parser.add_argument("output_dir")
args = parser.parse_args()

# mesh resolution
resolution = (float)(args.resolution)
r = 1
R = 2
c_r = [0, 0]
c_R = [0, 0]
# M is the number of times the slice will be replicated
M = 1
N = 16
theta = 2 * np.pi / N
# N = int(np.round(r * np.pi / resolution))

output_dir = args.output_dir
input_dir = args.input_dir
mesh_slice_file = input_dir + "/mesh.msh"

print(f'mesh_slice_file: {mesh_slice_file}')

# Load the mesh slice
mesh = meshio.read(mesh_slice_file)


print('********** Mesh before mirroring: **********')
msh.print_mesh_element_types(mesh)
msh.print_mesh_triangles(mesh)
msh.print_mesh_vertices(mesh)


# a curve representing the top line of the slice

# initialize the loop over 0 <= theta < 2 pi
r_1 = np.array([r, 0])
r_2 = cal.R(theta).dot(r_1)
r_4 = np.array([R, 0])
r_3 = cal.R(theta).dot(r_4)

print('Looping through circle ...')
for i in range(1, M+1):

    print(f'\t i = {i}')

    r_1 = np.copy(r_2)
    r_2 = cal.R(theta).dot(r_1)
    r_4 = np.copy(r_3)
    r_3 = cal.R(theta).dot(r_4)

    gamma_axis_of_symmetry = lambda t: cal.line(r_2, r_3, t)

    def point_on_axis_of_symmetry(point):
        return cal.point_on_line(point, gamma_axis_of_symmetry)

    def mirror_function(point):
        return cal.mirror_point_line(point, gamma_axis_of_symmetry)
print('... done.')

