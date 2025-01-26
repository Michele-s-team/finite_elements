'''
This code generates a 3d mesh given by a box and a sphere.
The mesh is given by a box with extremal points [0,0,0] , [L, B, H] to which we substract a sphere centered at c_r with radius r
We imagine looking at the mesh from a point at y=z=0 and x<0 and define left, right top bottom, fron and back edges accordingly

To see the values of subdomain_id assigned to each tagged element, see read_3dmeshg_box_ball.py and the comments in this file

run with
clear; clear; python3 generate_3dmesh_box_ball.py [resolution]
example:
clear; clear; rm -r solution; mkdir solution; python3 generate_3dmesh_box_ball.py 0.1
'''

import meshio
import argparse
import numpy as np
import gmsh
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()

warnings.filterwarnings("ignore")
gmsh.initialize()

gmsh.model.add("my model")
L = 1.0
B = 1.0
H = 1.0
c_r = [L/2.0, B/2.0, H/2.0]
r = 0.25
resolution = (float)(args.resolution)
print(f"Mesh resolution = {resolution}")

channel = gmsh.model.occ.addBox(0, 0, 0, L, B, H)
sphere = gmsh.model.occ.addSphere( c_r[0], c_r[1], c_r[2], r)
fluid = gmsh.model.occ.cut( [(3, channel)], [(3, sphere)] )

gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)

assert volumes == fluid[0]
#these is is the subdomain_id with which the volume [box-sphere] will be read in read_3dmesh_box_ball.py
box_minus_ball_subdomain_id = 8
gmsh.model.addPhysicalGroup( volumes[0][0], [volumes[0][1]], box_minus_ball_subdomain_id )
gmsh.model.setPhysicalName( volumes[0][0], box_minus_ball_subdomain_id, "box_minus_sphere" )



surfaces = gmsh.model.occ.getEntities(dim=2)
#these are the subdomain_ids with which the components will be read in read_3dmesh_box_ball.py
boundary_le_subdomain_id = 1
boundary_ri_subdomain_id = 2
boundary_fr_subdomain_id = 3
boundary_ba_subdomain_id = 4
boundary_to_subdomain_id = 5
boundary_bo_subdomain_id = 6
sphere_subdomain_id = 7

obstacles = []

#loop through all surfaces and tag them
for surface in surfaces:
    #compute the center of mass of each surface, and recognize according to the coordinates of the center of mass
    center_of_mass = gmsh.model.occ.getCenterOfMass( surface[0], surface[1] )

    if np.allclose( center_of_mass, [0, B / 2, H / 2] ):
        # the center of mass is close to [0, B / 2, H / 2] -> the surface under consideration is  boundary_le
        gmsh.model.addPhysicalGroup( surface[0], [surface[1]], boundary_le_subdomain_id )
        inlet = surface[1]
        gmsh.model.setPhysicalName( surface[0], boundary_le_subdomain_id, "boundary_le" )

    if (np.allclose( center_of_mass, [L, B / 2, H / 2] )):
        # the center of mass is close to [L, B / 2, H / 2] -> the surface under consideration is  boundary_ri
        gmsh.model.addPhysicalGroup( surface[0], [surface[1]], boundary_ri_subdomain_id )
        gmsh.model.setPhysicalName( surface[0], boundary_ri_subdomain_id, "boundary_ri" )

    # the center of mass is not the inlet nor the outlet:
    if (np.allclose( center_of_mass, [L / 2, 0, H / 2] )):
            # the y coordinate of the center of mass is 0 -> the surface under consideration is boundary_fr
            gmsh.model.addPhysicalGroup( surface[0], [surface[1]], boundary_fr_subdomain_id )
            gmsh.model.setPhysicalName( surface[0], boundary_fr_subdomain_id, "boundary_fr" )

    if (np.allclose( center_of_mass, [L / 2, B, H / 2] )):
        # the y coordinate of the center of mass is B -> the surface under consideration is boundary_ba
        gmsh.model.addPhysicalGroup( surface[0], [surface[1]], boundary_ba_subdomain_id )
        gmsh.model.setPhysicalName( surface[0], boundary_ba_subdomain_id, "boundary_ba" )

    if (np.allclose( center_of_mass, [L / 2, B / 2, H] )):
        # the z coordinate of the center of mass is H -> the surface under consideration is boundary_to
        gmsh.model.addPhysicalGroup( surface[0], [surface[1]], boundary_to_subdomain_id )
        gmsh.model.setPhysicalName( surface[0], boundary_to_subdomain_id, "boundary_to" )

    if (np.allclose( center_of_mass, [L / 2, B / 2, 0] )):
        # the z coordinate of the center of mass is 0 -> the surface under consideration is boundary_bo
        gmsh.model.addPhysicalGroup( surface[0], [surface[1]], boundary_bo_subdomain_id )
        gmsh.model.setPhysicalName( surface[0], boundary_bo_subdomain_id, "boundary_bo" )

    if (np.allclose( center_of_mass, c_r )):
        # the center of mass is c_r -> the surface under consideration is the sphere
        gmsh.model.addPhysicalGroup( surface[0], [surface[1]], sphere_subdomain_id )
        gmsh.model.setPhysicalName( surface[0], sphere_subdomain_id, "sphere" )




#set the resolution close to the obstacle
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

#set the resolution close to the inlet
inlet_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])

inlet_thre = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin",  resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax",  resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write("solution/mesh.msh")


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={
                           cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh


mesh_from_file = meshio.read("solution/mesh.msh")

#create a tetrahedron mesh in which the solid objects (volumes) will be stored
tetrahedron_mesh = create_mesh(mesh_from_file, "tetra", True)
meshio.write("solution/tetrahedron_mesh.xdmf", tetrahedron_mesh)

#create a triangle mesh in which the surfaces will be stored
triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write("solution/triangle_mesh.xdmf", triangle_mesh)