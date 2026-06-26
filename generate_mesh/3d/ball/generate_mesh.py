'''
This code generates a 3d mesh given by a ball

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/ball"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/ball/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import meshio
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam


# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file = output_directory + "mesh.msh"

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'

volume_id = 1
surface_id = 2
line_id = 3


geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

# add a volume object (a ball):
ball = model.add_ball(rpam.parameters["c_r"], rpam.parameters["r"], mesh_size=rpam.parameters["resolution"])

# add a line object
points = [model.add_point((0, 0, 0), mesh_size=rpam.parameters["resolution"]),
          model.add_point((0.2, 0.2, 0.2), mesh_size=rpam.parameters["resolution"])
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

# msh.print_mesh_edges_to_csv(mesh_file, rarg.args.output_directory + '/line_vertices.csv')
#
# # create a tetrahedron mesh (containing solid objects such as a ball)
# tetra_mesh = msh.create_mesh(mesh_from_file, "tetra", False)
# meshio.write(args.output_directory + "/tetra_mesh.xdmf", tetra_mesh)
#
# # create a triangle mesh (containing surfaces such as the ball surface): note that this will work only if some surfaces are present in the model
# triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", False)
# meshio.write(args.output_directory + "/triangle_mesh.xdmf", triangle_mesh)
#
# # print the mesh vertices to file
# mesh = msh.read_mesh(args.output_directory + "/tetra_mesh.xdmf")
# io.print_mesh_vertices_to_csv(mesh, args.output_directory + "/vertices.csv")

msh.full_write(mesh_file, ['tetra', 'triangle'], metadata, output_directory, False)

model.__exit__()
