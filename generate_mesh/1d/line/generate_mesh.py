'''
generate a  mesh given by a line
because mesh cannot be written and read properly when written on xdmf files, this 1d mesh is written to h5 files

Run it with
    python3 generate_half_circle_with_line_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/1d/line"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

from fenics import *
import math
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh
import runtime_arguments_generate_mesh as rarg
import read_parameters_generate_mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

output_directory = io.add_trailing_slash(rarg.args.output_directory)
print("output_directory = ", output_directory)

metadata = rpam.parameters.copy()
metadata['file_format'] = 'h5'


mesh_temp = IntervalMesh(int((rpam.parameters['x_r'] - rpam.parameters['x_l']) / rpam.parameters['resolution']), rpam.parameters['x_l'], rpam.parameters['x_r'])

# create a function for the lines
cell_function_temp = MeshFunction("size_t", mesh_temp, mesh_temp.topology().dim())
cell_function_temp.set_all(rpam.parameters['line_id'])  # Tag entire line as region parameters['line_id']

# creat a function for the vertices
vertex_function_temp = MeshFunction("size_t", mesh_temp, mesh_temp.topology().dim() - 1)
for vertex in vertices(mesh_temp):
    x = vertex.point().x()  # Get x-coordinate

    if math.isclose(x, rpam.parameters['x_l']):
        vertex_function_temp[vertex] = rpam.parameters['vertex_l_id']

    if math.isclose(x, rpam.parameters['x_r']):
        vertex_function_temp[vertex] = rpam.parameters['vertex_r_id']

'''
write the mesh lines and vertices to .h5 files: 
one needs to write them to .h5 file rather than to .xdmf file because only .h5 file can be properly read later on
'''
msh.write_mesh_components_h5(mesh_temp, output_directory + "line_mesh.h5", cell_function_temp, "cf")
msh.write_mesh_components_h5(mesh_temp, output_directory + "vertex_mesh.h5", vertex_function_temp, "vf")

# print mesh metadata
io.write_parameters_to_csv_file(output_directory + "mesh_metadata.csv", metadata)
