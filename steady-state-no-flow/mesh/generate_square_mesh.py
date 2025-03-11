'''
generate a mesh given by a square with a circular hole in it

run it with
python3 generate_square_mesh.py [resolution] [output directory]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_square_mesh.py 0.1 $SOLUTION_PATH

'''

import meshio
import gmsh
import pygmsh
import argparse

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument( "resolution" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

# mesh resolution
resolution = (float)( args.resolution )

# add '/' to output_directory if it is missing
output_directory = args.output_directory
output_directory = io.add_trailing_slash( output_directory )

mesh_file = output_directory + "mesh.msh"

# Channel parameters
# CHANGE PARAMETERS HERE
L = 0.5
h = L
r = 0.05
c_r = [L / 2.0, h / 2.0, 0]
# CHANGE PARAMETERS HERE


print( "L = ", L )
print( "h = ", h )
print( "r = ", r )
print( "c_r = ", c_r )
print( "resolution = ", resolution )

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

my_points = [model.add_point( (0, 0, 0), mesh_size=resolution ),
             model.add_point( (L, 0, 0), mesh_size=resolution ),
             model.add_point( (L, h, 0), mesh_size=resolution ),
             model.add_point( (0, h, 0), mesh_size=resolution )]

# Add lines between all points creating the rectangle
channel_lines = [model.add_line( my_points[i], my_points[i + 1] )
                 for i in range( -1, len( my_points ) - 1 )]

channel_loop = model.add_curve_loop( channel_lines )

circle_r = model.add_circle( c_r, r, mesh_size=resolution )

plane_surface = model.add_plane_surface( channel_loop, holes=[circle_r.curve_loop] )

model.synchronize()

model.add_physical( [plane_surface], "Volume" )
model.add_physical( [channel_lines[0]], "i" )
model.add_physical( [channel_lines[2]], "o" )
model.add_physical( [channel_lines[3]], "t" )
model.add_physical( [channel_lines[1]], "b" )
model.add_physical( circle_r.curve_loop.curves, "c" )

geometry.generate_mesh( dim=2 )
gmsh.write( mesh_file )

msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

gmsh.clear()
geometry.__exit__()

mesh_from_file = meshio.read( mesh_file )

line_mesh = msh.create_mesh( mesh_from_file, "line", prune_z=True )
meshio.write( output_directory + "line_mesh.xdmf", line_mesh )

triangle_mesh = msh.create_mesh( mesh_from_file, "triangle", prune_z=True )
meshio.write( output_directory +  "triangle_mesh.xdmf", triangle_mesh )


# print the mesh vertices to file
mesh = msh.read_mesh( output_directory + "triangle_mesh.xdmf" )
io.print_vertices_to_csv_file( mesh, output_directory + "vertices.csv" )