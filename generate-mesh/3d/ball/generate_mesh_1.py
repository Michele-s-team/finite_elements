'''
This code generates a 3d mesh with the shape of a ball (filled inside)

run with
clear; clear; python3 generate_mesh_1.py [resolution]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_1.py 0.1 $SOLUTION_PATH

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
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

#mesh resolution
resolution = (float)(args.resolution)
print("resolution = ", resolution)

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

#generate a ball
#CHANGE PARAMETERS HERE
r = 1.0
c_r = [0, 0, 0]
#CHANGE PARAMETERS HERE

print( "r = ", r )
print( "c_r = ", c_r )

ball = model.add_ball(c_r, r,  mesh_size=resolution)
model.synchronize()
model.add_physical([ball], "ball")

geometry.generate_mesh(dim=3)
gmsh.write("solution/mesh.msh")
model.__exit__()


mesh_from_file = meshio.read("solution/mesh.msh")

#create a tetrahedron mesh
tetrahedron_mesh = msh.create_mesh(mesh_from_file, "tetra", True)
meshio.write(args.output_directory + "/tetrahedron_mesh.xdmf", tetrahedron_mesh)
