'''
run with
clear; clear; python3 generate_3dmesh.py [resolution]
example:
clear; clear; rm -r solution; mkdir solution;  python3 generate_3dmesh.py 0.1
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

#mesh parameters
#CHANGE PARAMETERS HERE
r = 1.0
c_r = [0, 0, 0]
#CHANGE PARAMETERS HERE

print("r = ", r)
print("c_r = ", c_r)
print("resolution = ", resolution)


# Initialize empty geometry using the build in kernel in GMSH
geometry = pygmsh.occ.Geometry()
# Fetch model we would like to add data to
model = geometry.__enter__()


#add a 3d object:
ball = model.add_ball(c_r, r,  mesh_size=resolution)



model.synchronize()

#add the volume object (ball), which will be added with subdomain_id = 2
model.add_physical([ball], "ball")

#add the 2d objet (ball surface, i.e., sphere): find out the sphere surface and add it to the model
#this is name_to_read which will be shown in paraview and subdomain_id which will be used in the code which reads the mesh in `ds_custom = Measure("ds", domain=mesh, subdomain_data=sf, subdomain_id=1)`
'''
the  surfaces are tagged with the following subdomain_ids: 

If you have a doubt about the subdomain_ids, see name_to_read in tetrahedron_mesh.xdmf with Paraview
'''
sphere_tag = 1
dim_facet = 2 # for facets in 3D
sphere_boundaries = []
volumes = gmsh.model.getEntities( dim=3 )
#extract the boundaries from the mesh
boundaries = gmsh.model.getBoundary(volumes, oriented=False)
#add the surface objects: loop through the surfaces in the model and add them as physical objects: here the sphere surface will be added with subdomain_id = 1
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

#create a tetrahedron mesh (containing solid objects such as a ball)
tetrahedron_mesh = create_mesh(mesh_from_file, "tetra", True)
meshio.write("solution/tetrahedron_mesh.xdmf", tetrahedron_mesh)

#create a triangle mesh (containing surfaces such as the ball surface): note that this will work only if some surfaces are present in the model
triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write("solution/triangle_mesh.xdmf", triangle_mesh)

'''
#create a line mesh
line_mesh = create_mesh(mesh_from_file, "line", True)
meshio.write("solution/line_mesh.xdmf", line_mesh)

#create a vertex mesh
vertex_mesh = create_mesh(mesh_from_file, "vertex", True)
meshio.write("solution/vertex_mesh.xdmf", vertex_mesh)
'''