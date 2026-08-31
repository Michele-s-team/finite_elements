'''
This code generates a 2d mesh (A) given by a square with an arbitrary shape (the shape is meshed inside), plus a one-dimensional mesh (B) given by a line. The line mesh B corresponds to the shape boundary of A, stretched on a line. 

Here 
    - A is mesh_0
    - B is mesh_1
    
A has 2 sub_meshes 
    - sub_mesh_0_0
    - sub_mesh_0_1
    
and B has no sub_meshes

Run it with
    python3 generate_mesh.py [path where to read rpam.parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

shape_coordinates = None

if rpam.parameters['shape_format'] == 'coordinates':
    # the  shape is provided directly as a sequence of coordinates of the shape points -> set shape_coordinates to these coordinates

    print('The shape is provided as a set of coordinates.')

    shape_coordinates = rpam.parameters['shape_coordinates']

elif rpam.parameters['shape_format'] == 'parametric':
    #  the shape is a given, parametric geometrical shape, and it is provided in terms of the parameters of this shape

    shape_parametric_form = io.read_function_expresssion(rpam.parameters['shape_parametric_form'])
    shape_coordinates = [shape_parametric_form(i/rpam.parameters['N']) for i in range(rpam.parameters['N'])]

msh.generate_square_shape_line_mesh(shape_coordinates, rarg.args.parameter_directory, rarg.args.output_directory)

