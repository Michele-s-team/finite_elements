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
M = 2
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
