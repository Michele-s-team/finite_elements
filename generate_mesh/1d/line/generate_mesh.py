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

msh.genereate_line_mesh(rpam.parameters['x_l'], rpam.parameters['x_r'], int((rpam.parameters['x_r'] - rpam.parameters['x_l']) / rpam.parameters['resolution']),
                        rpam.parameters['line_id'], rpam.parameters['vertex_l_id'], rpam.parameters['vertex_r_id'],
                        output_directory, metadata)
