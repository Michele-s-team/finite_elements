'''
This code generates a  mesh given by a slice of a ring

Run with
    python3 generate_mesh_ring_slice.py [mesh resolution] [path where to store the mesh]

Example:
    clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_ring_slice.py 0.1 $SOLUTION_PATH
'''

import argparse
from fenics import *
import meshio
import numpy as np
import sys

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
c_r = [0, 0]
c_R = [0, 0]
# the angular witdh of the slice is 2 \pi/N = theta
N = 8
theta = 2 * np.pi / N

output_dir = args.output_dir
mesh_file = output_dir + "/mesh.msh"

msh.generate_mesh_ring_slice(r, R, c_r, c_R, theta, resolution, mesh_file)

# Load the half-mesh
mesh = meshio.read(mesh_file)

line_mesh = msh.create_mesh(mesh, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

triangle_mesh = msh.create_mesh(mesh, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_dir + "/triangle_mesh.xdmf")
io.print_vertices_to_csv_file(mesh, output_dir + "/vertices.csv")
