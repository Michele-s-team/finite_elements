'''
This code generates a 3d mesh given by a box and a sphere

To see the values of subdomain_id assigned to each tagged element, see read_3dmeshg_box_ball.py and the comments in this file

run with
clear; clear; python3 generate_3dmesh_box_ball.py [resolution]
example:
clear; clear; rm -r solution; mkdir solution; python3 generate_3dmesh_box_ball.py 0.1
'''

import meshio
import gmsh
import pygmsh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("resolution")
args = parser.parse_args()

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
'''
the  surfaces are tagged with the following subdomain_ids: 
- left: subdomain_id = 1
- right: subdomain_id = 6
- front: subdomain_id = 2
- back: subdomain_id = 4
- top: subdomain_id = 3
- bottom: subdomain_id = 5
- sphere: subdomain_id = 7
If you have a doubt about the subdomain_ids, see name_to_read in tetrahedron_mesh.xdmf with Paraview
'''
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