'''
generate a mesh given by a square with a circular hole in it

run it with
python3 generate_square_mesh.py [resolution] [output directory]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_ring_mesh.py 0.1 $SOLUTION_PATH

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

# parameters
r = 1.0
R = 2.0
c_r = [0, 0, 0]
c_R = [0, 0, 0]

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()

# Add circle
circle_r = model.add_circle( c_r, r, mesh_size=resolution )
circle_R = model.add_circle( c_R, R, mesh_size=resolution )

plane_surface = model.add_plane_surface( circle_R.curve_loop, holes=[circle_r.curve_loop] )

model.synchronize()
model.add_physical( [plane_surface], "Volume" )

# I will read this tagged element with `ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)`
model.add_physical( circle_r.curve_loop.curves, "Circle r" )
model.add_physical( circle_R.curve_loop.curves, "Circle R" )

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
