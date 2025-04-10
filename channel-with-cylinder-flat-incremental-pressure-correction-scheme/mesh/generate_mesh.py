'''
generate a mesh given by a rectangle with a circular hole

run it with
python3 generate_mesh.py [resolution] [output directory]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py 0.1 $SOLUTION_PATH
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
L = 2.2
h = 0.41
r = 0.05
c_r = [0.2, 0.2, 0]

print( "L = ", L )
print( "h = ", h )
print( "r = ", r )
print( "c_r = ", c_r )
print( "resolution = ", resolution )

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

# Add circle
circle_r = model.add_circle( c_r, r, mesh_size=resolution )
rectangle_Lh = model.add_rectangle( 0, L, 0, h, 0, mesh_size=resolution )

# Add points with finer resolution on left side
points = [model.add_point( (0, 0, 0), mesh_size=resolution ),
          model.add_point( (L, 0, 0), mesh_size=resolution ),
          model.add_point( (L, h, 0), mesh_size=resolution ),
          model.add_point( (0, h, 0), mesh_size=resolution )]

# Add lines between all points creating the rectangle
channel_lines = [model.add_line( points[i], points[i + 1] )
                 for i in range( -1, len( points ) - 1 )]

# Create a line loop and plane surface for meshing
channel_loop = model.add_curve_loop( channel_lines )

# plane_surface = model.add_plane_surface(     rectangle_Lh.curve_loop, holes=[circle_r.curve_loop])
plane_surface = model.add_plane_surface( channel_loop, holes=[circle_r.curve_loop] )

## Call gmsh kernel before add physical entities
model.synchronize()

# The final step before mesh generation is to mark the different boundaries and the volume mesh. Note that with pygmsh, boundaries with the same tag has to be added simultaneously. In this example this means that we have to add the top and
#  bottom wall in one function call.

# volume_marker = 6
model.add_physical( [plane_surface], "Volume" )
model.add_physical( [channel_lines[0]], "Inflow" )
model.add_physical( [channel_lines[2]], "Outflow" )
model.add_physical( [channel_lines[3]], "Top" )
model.add_physical( [channel_lines[1]], "Bottom" )
model.add_physical( circle_r.curve_loop.curves, "Obstacle" )

# We generate the mesh using the pygmsh function `generate_mesh`. Generate mesh returns a `meshio.Mesh`. However, this mesh is tricky to extract physical tags from. Therefore we write the mesh to file using the `gmsh.write` function.

geometry.generate_mesh( 64 )
gmsh.write( mesh_file )
msh.write_mesh_to_csv( mesh_file, output_directory + 'line_vertices.csv' )

gmsh.clear()
geometry.__exit__()

mesh_from_file = meshio.read( mesh_file )

line_mesh = msh.create_mesh( mesh_from_file, "line", prune_z=True )
meshio.write( output_directory + "line_mesh.xdmf", line_mesh )

triangle_mesh = msh.create_mesh( mesh_from_file, "triangle", prune_z=True )
meshio.write( output_directory + "triangle_mesh.xdmf", triangle_mesh )

# print the mesh vertices to file
mesh = msh.read_mesh( output_directory + "triangle_mesh.xdmf" )
io.print_vertices_to_csv_file( mesh, output_directory + "vertices.csv" )
