'''
This code generates a 3d mesh given by a ball

Run with
    clear; clear; python3 generate_ball_mesh.py [resolution] [output directory]
Example:
    clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_ball_mesh.py 0.1 $SOLUTION_PATH
'''

import argparse
import gmsh
import meshio
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

mesh_file = args.output_directory + "/mesh.msh"

volume_id = 1
surface_id = 2
line_id = 3

# mesh resolution
resolution = (float)(args.resolution)

# mesh parameters
# CHANGE PARAMETERS HERE
r = 1.0
c_r = [0, 0, 0]
# CHANGE PARAMETERS HERE

print("r = ", r)
print("c_r = ", c_r)
print("resolution = ", resolution)

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a volume object (a ball):
ball = model.add_ball(c_r, r, mesh_size=resolution)

# add a line object
points = [model.add_point((0, 0, 0), mesh_size=resolution),
          model.add_point((0.2, 0.2, 0.2), mesh_size=resolution)
          ]
line = [model.add_line(points[0], points[1])]

model.synchronize()

# tag 3d objects
volumes = gmsh.model.getEntities(dim=3)
for volume in volumes:
    gmsh.model.addPhysicalGroup(3, [volume[1]], volume_id)  # Tag 1 for volume
    gmsh.model.setPhysicalName(3, volume_id, "volume")

# tag 2d objects
boundary_dimension = 2  # for facets in 3D
boundaries = gmsh.model.getBoundary(volumes, oriented=False)
gmsh.model.addPhysicalGroup(boundary_dimension, [boundary[1] for boundary in boundaries], surface_id)  # Tag 1 for volume
gmsh.model.setPhysicalName(boundary_dimension, surface_id, "surface")

geometry.generate_mesh(dim=3)
gmsh.write(mesh_file)
mesh_from_file = meshio.read(mesh_file)

msh.print_mesh_lines_to_csv(mesh_file, args.output_directory + '/line_vertices.csv')

model.__exit__()

# create a tetrahedron mesh (containing solid objects such as a ball)
tetrahedron_mesh = msh.create_mesh(mesh_from_file, "tetra", False)
meshio.write(args.output_directory + "/tetrahedron_mesh.xdmf", tetrahedron_mesh)

# create a triangle mesh (containing surfaces such as the ball surface): note that this will work only if some surfaces are present in the model
triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", False)
meshio.write(args.output_directory + "/triangle_mesh.xdmf", triangle_mesh)

'''
#create a line mesh
line_mesh = create_mesh(mesh_from_file, "line", True)
meshio.write(args.output_directory + "/line_mesh.xdmf", line_mesh)

#create a vertex mesh
vertex_mesh = create_mesh(mesh_from_file, "vertex", True)
meshio.write(args.output_directory + "/vertex_mesh.xdmf", vertex_mesh)
'''

# print the mesh vertices to file
mesh = msh.read_mesh(args.output_directory + "/tetrahedron_mesh.xdmf")
io.print_vertices_to_csv_file(mesh, args.output_directory + "/vertices.csv")
