'''
generate a mesh given by a square whose top line is a one-dimensional submesh

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
    
Here 'sub_mesh_0' is the two-dimensional square mesh and 'sub_mesh_1' is the one-dimensional top edge of the square.
'''

from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg

msh.generate_square_no_circle_curve_mesh([], rarg.args.parameter_directory, rarg.args.output_directory)