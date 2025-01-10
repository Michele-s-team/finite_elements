'''
This code generates a 3d mesh given by a box and a sphere

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
L, B, H, r = 2.5, 0.41, 0.41, 0.05
resolution = r / 10

channel = gmsh.model.occ.addBox(0, 0, 0, L, B, H)
cylinder = gmsh.model.occ.addCylinder(0.5, 0, 0.2, 0, B, 0, r)
fluid = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)])

gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)
assert volumes == fluid[0]
fluid_marker = 11
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid volume")

surfaces = gmsh.model.occ.getEntities(dim=2)
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 1, 3, 5, 7
walls = []
obstacles = []
for surface in surfaces:
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    if np.allclose(com, [0, B / 2, H / 2]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], inlet_marker)
        inlet = surface[1]
        gmsh.model.setPhysicalName(surface[0], inlet_marker, "Fluid inlet")
    elif np.allclose(com, [L, B / 2, H / 2]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], outlet_marker)
        gmsh.model.setPhysicalName(surface[0], outlet_marker, "Fluid outlet")
    elif (
        np.isclose(com[2], 0)
        or np.isclose(com[1], B)
        or np.isclose(com[2], H)
        or np.isclose(com[1], 0)
    ):
        walls.append(surface[1])
    else:
        obstacles.append(surface[1])
gmsh.model.addPhysicalGroup(2, walls, wall_marker)
gmsh.model.setPhysicalName(2, wall_marker, "Walls")
gmsh.model.addPhysicalGroup(2, obstacles, obstacle_marker)
gmsh.model.setPhysicalName(2, obstacle_marker, "Obstacle")

distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

inlet_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])

inlet_thre = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", 5 * resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", 10 * resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write("mesh3D.msh")





'''
#mesh resolution
resolution = (float)(args.resolution)
print("resolution = ", resolution)

# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()

#generate a box
#
#CHANGE PARAMETERS HERE
L1 = 1
L2 = 1
L3 = 1
#CHANGE PARAMETERS HERE
#generate a ball
#CHANGE PARAMETERS HERE
r = 0.25
c_r = [0.5, 0.5, 0.5]
#CHANGE PARAMETERS HERE

print( "r = ", r )
print( "c_r = ", c_r )

box = model.add_box([0, 0, 0], [L1, L2, L3], mesh_size=resolution)
#add a ball to the model
ball = model.add_ball(c_r, r,  mesh_size=resolution)

box_minus_ball = model.boolean_difference( box, ball )


model.synchronize()

#add the ball to the model
# model.add_physical( ball, "ball" )
# model.add_physical( box, "box" )
#the volume of {box minus ball} is written in tetrahedron_mesh.xdmf with subdomain_id = 8
model.add_physical( box_minus_ball, "difference" )

#find out the sphere surface and add it to the model
#this is name_to_read which will be shown in paraview and subdomain_id which will be used in the code which reads the mesh in `ds_custom = Measure("ds", domain=mesh, subdomain_data=sf, subdomain_id=1)`

sphere_tag = 1
dim_facet = 2 # for facets in 3D
sphere_boundaries = []
volumes = gmsh.model.getEntities( dim=3 )
#extract the boundaries from the mesh
boundaries = gmsh.model.getBoundary(volumes, oriented=False)
#loop through the boundaries
id=0
for boundary in boundaries:
    center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
    sphere_boundaries.append(boundary[1])
    gmsh.model.addPhysicalGroup(dim_facet, sphere_boundaries)
    print(f"surface # {id}, center of mass = {center_of_mass}")
    id+=1

geometry.generate_mesh(dim=3)
gmsh.write("solution/mesh.msh")
model.__exit__()

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
'''