'''
Ths code generates a 1d mesh given by a segment with a vertex in the segment

Run with
    clear; clear; python3 generate_mesh_line_vertex.py [resolution]
Example:
    clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh_line_vertex.py 0.1 $SOLUTION_PATH
'''


import meshio
import gmsh
import pygmsh
import argparse
import numpy as np

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

mesh_file = args.output_directory + "/mesh.msh"

#mesh resolution
resolution = (float)(args.resolution)

#CHANGE PARAMETERS HERE
L = 1.0
x_p = np.pi / 8.0
#CHANGE PARAMETERS HERE

print(f'L = {L}, x_p = {x_p}')
print("resolution = ", resolution)


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

#add a 1d object a set of lines
points = [model.add_point( (0, 0, 0), mesh_size=resolution ),
          model.add_point((x_p, 0, 0), mesh_size=resolution),
          model.add_point((L, 0, 0), mesh_size=resolution)
          ]
my_lines = [model.add_line( points[0], points[1] ), model.add_line( points[1], points[2] )]

#add a 2d object:  a plane surface starting from the 4 lines above


model.synchronize()

print("# of lines added = ", len(my_lines))

model.add_physical([my_lines[0]], "line1")
model.add_physical([my_lines[1]], "line2")
model.add_physical([points[0]], "point_l")
model.add_physical([points[2]], "point_r")
model.add_physical([points[1]], "point_in")

geometry.generate_mesh(dim=3)
gmsh.write(mesh_file)


model.__exit__()


mesh_from_file = meshio.read(mesh_file)

'''
#create a tetrahedron mesh
tetrahedron_mesh = create_mesh(mesh_from_file, "tetra", True)
meshio.write("solution/tetrahedron_mesh.xdmf", tetrahedron_mesh)

#create a triangle mesh
triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write("solution/triangle_mesh.xdmf", triangle_mesh)
'''

#create a line mesh
line_mesh = msh.create_mesh(mesh_from_file, "line", True)
meshio.write(args.output_directory +  "/line_mesh.xdmf", line_mesh)

#create a vertex mesh
vertex_mesh = msh.create_mesh(mesh_from_file, "vertex", True)
meshio.write(args.output_directory +  "/vertex_mesh.xdmf", vertex_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh( args.output_directory + "/line_mesh.xdmf" )
io.print_vertices_to_csv_file( mesh, args.output_directory + "/vertices.csv" )
