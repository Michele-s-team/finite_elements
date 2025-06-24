'''
This code generates a 3d mesh given by a box

Run it with
    python3 generate_box_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/box"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/box/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_box_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import argparse
import gmsh
import numpy as np
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

# mesh resolution
resolution = (float)(args.resolution)
print("resolution = ", resolution)

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# CHANGE PARAMETERS HERE
L = [3, 2, 1]

volume_id = 1
boundary_le_id = 2
boundary_ri_id = 3
boundary_to_id = 4
boundary_bo_id = 5
boundary_fr_id = 6
boundary_ba_id = 7
# CHANGE PARAMETERS HERE

print(f'L = {L}\nresolution = {resolution}')

box = model.add_box([0, 0, 0], [L[0], L[1], L[2]], mesh_size=resolution)

model.synchronize()

# tag 3d objects
volumes = gmsh.model.getEntities(dim=3)
for volume in volumes:
    gmsh.model.addPhysicalGroup(3, [volume[1]], volume_id)  # Tag 1 for volume
    gmsh.model.setPhysicalName(3, volume_id, "volume")

# tag 2d objects
surfaces = gmsh.model.occ.getEntities(dim=2)

for surface in surfaces:
    # compute the center of mass of each surface, and recognize according to the coordinates of the center of mass
    center_of_mass = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])

    if np.isclose(center_of_mass[0], 0):
        # the x coordinate of the center of mass is close to  0 -> I am on boundary_l
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundary_le_id)
        gmsh.model.setPhysicalName(surface[0], boundary_le_id, "boundary_le")

    if np.isclose(center_of_mass[0], L[0]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundary_ri_id)
        gmsh.model.setPhysicalName(surface[0], boundary_ri_id, "boundary_ri")

    if np.isclose(center_of_mass[1], 0):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundary_bo_id)
        gmsh.model.setPhysicalName(surface[0], boundary_bo_id, "boundary_bo")

    if np.isclose(center_of_mass[1], L[1]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundary_to_id)
        gmsh.model.setPhysicalName(surface[0], boundary_to_id, "boundary_to")

    if np.isclose(center_of_mass[2], 0):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundary_ba_id)
        gmsh.model.setPhysicalName(surface[0], boundary_ba_id, "boundary_ba")

    if np.isclose(center_of_mass[2], L[2]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundary_fr_id)
        gmsh.model.setPhysicalName(surface[0], boundary_fr_id, "boundary_fr")

geometry.generate_mesh(dim=3)
gmsh.write(mesh_file)

msh.print_mesh_lines_to_csv(mesh_file, args.output_directory + '/line_vertices.csv')

model.__exit__()

mesh_from_file = meshio.read(mesh_file)

# create a tetrahedron mesh
tetra_mesh = msh.create_mesh(mesh_from_file, "tetra", False)
meshio.write(args.output_directory + "/tetra_mesh.xdmf", tetra_mesh)

# create a triangle mesh (containing surfaces such as the ball surface): note that this will work only if some surfaces are present in the model
triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", False)
meshio.write(args.output_directory + "/triangle_mesh.xdmf", triangle_mesh)

# print the mesh vertices to file
mesh = msh.read_mesh(args.output_directory + "/tetra_mesh.xdmf")
io.print_mesh_vertices_to_csv(mesh, args.output_directory + "/vertices.csv")
