'''
This code generates a 3d mesh given by a box

run with
clear; clear; python3 generate_3dmesh_box.py [resolution]
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

import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument( "resolution" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

# mesh resolution
resolution = (float)( args.resolution )
print( "resolution = ", resolution )

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

# generate a box
#
# CHANGE PARAMETERS HERE
L1 = 1
L2 = 0.5
L3 = 0.45
# CHANGE PARAMETERS HERE
box = model.add_box( [0, 0, 0], [L1, L2, L3], mesh_size=resolution )
model.synchronize()
model.add_physical( [box], "box" )
#


geometry.generate_mesh( dim=3 )
gmsh.write( "solution/mesh.msh" )
model.__exit__()

mesh_from_file = meshio.read( args.output_directory + "/mesh.msh" )

# create a tetrahedron mesh
tetrahedron_mesh = msh.create_mesh( mesh_from_file, "tetra", True )
meshio.write( args.output_directory + "/tetrahedron_mesh.xdmf", tetrahedron_mesh )
