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

import input_output as io
import mesh.utils as msh
import parameters.read.mesh as rpam
import runtime_arguments_generate_mesh as rarg

curve_coordinates = None

if rpam.parameters['curve_format'] == 'coordinates':
    # the  curve is provided directly as a sequence of coordinates of the curve points -> set curve_coordinates to these coordinates

    print('The curve is provided as a set of coordinates.')

    curve_coordinates = rpam.parameters['curve_coordinates']

elif rpam.parameters['curve_format'] == 'parametric':
    #  the curve is a given, parametric geometrical curve, and it is provided in terms of the parameters of this curve

    curve_parametric_form = io.read_function_expresssion(rpam.parameters['curve_parametric_form'])
    curve_coordinates = [curve_parametric_form(i/rpam.parameters['N']) for i in range(rpam.parameters['N'])]

msh.generate_square_no_circle_curve_mesh(curve_coordinates, rarg.args.parameter_directory, rarg.args.output_directory)