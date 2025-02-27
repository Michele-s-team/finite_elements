'''
run with
clear; clear; python3 generate_mesh.py [resolution]
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
parser.add_argument("resolution")
parser.add_argument("output_directory")
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

#add a volume object (a ball):
ball = model.add_ball(c_r, r,  mesh_size=resolution)

#add a line object
points = [model.add_point( (0, 0, 0), mesh_size=resolution ),
          model.add_point((0.2, 0.2, 0.2), mesh_size=resolution)
          ]
line = [model.add_line( points[0], points[1] )]



model.synchronize()
volumes = gmsh.model.getEntities( dim=3 )

#tag the volume object (ball), which will be added with subdomain_id = 2
model.add_physical([ball], "ball")

#tag the surface objet (ball surface, i.e., sphere): find out the sphere surface and add it to the model
#this is name_to_read which will be shown in paraview and subdomain_id which will be used in the code which reads the mesh in `ds_custom = Measure("ds", domain=mesh, subdomain_data=sf, subdomain_id=1)`
'''
the  surfaces are tagged with the following subdomain_ids: 
If you have a doubt about the subdomain_ids, see name_to_read in triangle_mesh.xdmf with Paraview
'''
dim_facet = 2 # for facets in 3D
sphere_boundaries = []
#extract the boundaries from the mesh
boundaries = gmsh.model.getBoundary(volumes, oriented=False)
#add the surface objects: loop through the surfaces in the model and add them as physical objects: here the sphere surface will be added with subdomain_id = 1
print("*********** surfaces : ***********  ", boundaries)
id=0
for boundary in boundaries:
    center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
    sphere_boundaries.append(boundary[1])
    gmsh.model.addPhysicalGroup(dim_facet, sphere_boundaries)
    print(f"surface # {id}, center of mass = {center_of_mass}")
    id+=1


# print("dir = " , dir(gmsh.model))


#tag the line objet : find out the line  and add it to the model
# the lines are tagged with the following subdomain_ids:
# If you have a doubt about the subdomain_ids, see name_to_read in line_mesh.xdmf with Paraview

dim_segment = 1 # for lines in 3D
#extract the segments from the mesh
segments = gmsh.model.occ.getEntities(dim=1)

#add the segment objects: loop through the segments in the model and add them as physical objects: here line[0] surface will be added with subdomain_id =
print("*********** segments : *********** ", segments)
id=0
line_segments = []
for segment in segments:
    center_of_mass = gmsh.model.occ.getCenterOfMass(segment[0], segment[1])
    line_segments.append(segment[1])
    # gmsh.model.addPhysicalGroup(dim_segment, line_segments)
    print(f"segment # {id}, center of mass = {center_of_mass}, segment: {segment}")
    id+=1



geometry.generate_mesh(dim=3)
gmsh.write(args.output_directory + "/mesh.msh")
model.__exit__()




mesh_from_file = meshio.read(args.output_directory + "/mesh.msh")

#create a tetrahedron mesh (containing solid objects such as a ball)
tetrahedron_mesh = msh.create_mesh(mesh_from_file, "tetra", False)
meshio.write(args.output_directory + "/tetrahedron_mesh.xdmf", tetrahedron_mesh)

#create a triangle mesh (containing surfaces such as the ball surface): note that this will work only if some surfaces are present in the model
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