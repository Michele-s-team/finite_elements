'''
generate a mesh given by a square with a polygon-shaped hole in it

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')
print(f'output_directory = "{rarg.args.output_directory}"')

polygon_coordinates = None

if rpam.parameters['polygon_format'] == 'coordinates':
    # the polygon shape is provided directly as a sequence of coordinates of the polygon points -> set polygon_coordinates to these coordinates

    polygon_coordinates = rpam.parameters['polygon_coordinates']

else:
    #  the polygon shape is a given, parameteric geometrical shape, and it is provided in terms of the parameters of this shape

    if rpam.parameters['polygon_format'] == 'ellipse':
    # the polygon shape is an ellipse -> obtain polygon_cordinates from the ellipse parameters

        polygon_coordinates = cal.points_ellipse(rpam.parameters['a'], rpam.parameters['b'], rpam.parameters['c'], rpam.parameters['N'])

    # here you can have other cases corresponding to other geometrical shapes (circle, etc... )


msh.generate_square_polygon_mesh(polygon_coordinates, rarg.args.parameter_directory, rarg.args.output_directory)
