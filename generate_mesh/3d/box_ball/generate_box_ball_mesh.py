'''
This code generates a 3d mesh given by a box with a spherical hole
The mesh is given by a box with extremal points [0,0,0] , L to which we subtract a sphere centered at c_r with radius r
We imagine looking at the mesh from a point at y=z=0 and x<0 and define left, right top bottom, from and back edges accordingly

Run with
    python3 generate_box_ball_mesh.py [resolution]
Example:
    SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_box_ball_mesh.py 0.1 $SOLUTION_PATH
'''

import argparse
import gmsh
import meshio
import numpy as np
import sys
import warnings

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

warnings.filterwarnings("ignore")
gmsh.initialize()

mesh_file = args.output_directory + "/mesh.msh"

gmsh.model.add("my model")

resolution = (float)(args.resolution)
print(f"Mesh resolution = {resolution}")

# CHANGE PARAMETERS HERE
L = [1, 0.95, 0.9]
c_r = [L[0] / 2.0, L[1] / 2.0, L[2] / 2.0]
r = 0.25

volume_id = 1
boundary_le_id = 2
boundary_ri_id = 3
boundary_to_id = 4
boundary_bo_id = 5
boundary_fr_id = 6
boundary_ba_id = 7
boundary_sphere_id = 8

sphere_resolution = resolution
# CHANGE PARAMETERS HERE



channel = gmsh.model.occ.addBox(0, 0, 0, L[0], L[1], L[2])
sphere = gmsh.model.occ.addSphere(c_r[0], c_r[1], c_r[2], r)
fluid = gmsh.model.occ.cut([(3, channel)], [(3, sphere)])

gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)

assert volumes == fluid[0]
# these is is the subdomain_id with which the volume [box-sphere] will be read in read_3dmesh_box_ball.py
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], volume_id)
gmsh.model.setPhysicalName(volumes[0][0], volume_id, "volume")

surfaces = gmsh.model.occ.getEntities(dim=2)

obstacles = []

# loop through all surfaces and tag them
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

    if (np.allclose(center_of_mass, c_r)):
        # the center of mass is c_r -> the surface under consideration is the sphere
        obstacles.append(surface[1])  # Save the tag of the sphere surface
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], boundary_sphere_id)
        gmsh.model.setPhysicalName(surface[0], boundary_sphere_id, "sphere")

# set the resolution close to the obstacle
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", sphere_resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", r)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2*r)

gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)

# create a tetrahedron mesh in which the solid objects (volumes) will be stored
tetrahedron_mesh = msh.create_mesh(mesh_from_file, "tetra", False)
meshio.write(args.output_directory + "/tetrahedron_mesh.xdmf", tetrahedron_mesh)

# create a triangle mesh in which the surfaces will be stored
triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write(args.output_directory + "/triangle_mesh.xdmf", triangle_mesh)
