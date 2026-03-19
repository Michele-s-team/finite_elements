'''
generate a mesh given by a square with a polygon-shaped hole in it

Here the polygon may be provided in two ways
    1. if 'polygon_format' == 'coordinates' in mesh_parameters.csv, the polygon is provided as a set of raw coordinates
    2. if 'polygon_format' == [name of a geometrical shape] in mesh_parameters.csv, the polygon is provided implicitly in terms of the parameters of that shape, which are provided into mesh_parameters.csv, for example:"
        2.1: 'polygon_format' == 'ellipse': then mesh_parameters.csv contains the ellipse semi-axes 'a', 'b', the center 'c', and the number of vertices 'N' which divide the ellipse boundary into segments. 
        2.2 ... other geometrical shapes are possible ... 

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

    print('The polygon shape is provided as a set of coordinates.')
    polygon_coordinates = rpam.parameters['polygon_coordinates']

else:
    #  the polygon shape is a given, parameteric geometrical shape, and it is provided in terms of the parameters of this shape

    if rpam.parameters['polygon_format'] == 'ellipse':
    # the polygon shape is an ellipse -> obtain polygon_cordinates from the ellipse parameters

        print('The polygon shape is an ellipse.')
        polygon_coordinates = cal.points_ellipse(rpam.parameters['a'], rpam.parameters['b'], rpam.parameters['c'], rpam.parameters['N'],
                                                 phi=rpam.parameters['phi'])

    # here you can have other cases corresponding to other geometrical shapes (circle, etc... )


msh.generate_square_polygon_mesh(polygon_coordinates, rarg.args.parameter_directory, rarg.args.output_directory)
