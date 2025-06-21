'''
This code generates a  mesh given by a slice of a ring

Run with
    clear; clear; python3 generate_mesh_ring_slice.py [path where to read the parameter file] [path where to store the solution]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/ring/ring_slice"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/1d/ring_slice/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_ring_slice.py $PARAMETERS_PATH $SOLUTION_PATH
'''

from fenics import *
import meshio
import numpy as np
import sys

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh
import runtime_arguments_generate_mesh as rarg
import read_parameters_generate_mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

output_dir = rarg.args.output_directory
mesh_file_name = output_dir + "/mesh.msh"
mesh_metadata_file_name = rarg.args.output_directory + '/mesh_metadata.csv'

# the angular width of the slice is 2 \pi/N = theta
theta = 2 * np.pi / rpam.parameters["N"]

msh.generate_mesh_ring_slice(rpam.parameters["r"], rpam.parameters["R"], rpam.parameters["c_r"], rpam.parameters["c_R"], theta, rpam.parameters["resolution"], mesh_file_name)

# Load the half-mesh
mesh = meshio.read(mesh_file_name)

# create a line mesh
line_mesh = msh.create_mesh(mesh, "line", prune_z=True)
meshio.write(output_dir + "/line_mesh.xdmf", line_mesh)

# create a triangle mesh
triangle_mesh = msh.create_mesh(mesh, "triangle", prune_z=True)
meshio.write(output_dir + "/triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(output_dir + "/triangle_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, output_dir + "/vertices.csv")

# print mesh metadata
io.write_parameters_to_csv_file(mesh_metadata_file_name, [('r', rpam.parameters['r']), \
                                                          ('R', rpam.parameters['R']), \
                                                          ('c_r', rpam.parameters['c_r']), \
                                                          ('c_R', rpam.parameters['c_R']), \
                                                          ('N', rpam.parameters['N']), \
                                                          ('resolution', rpam.parameters['resolution'])])
