'''
This code generates a 3d mesh given by a box with a spherical hole
The mesh is given by a box with extremal points [0,0,0] , L to which we subtract a sphere centered at c_r with radius r
We imagine looking at the mesh from a point at y=z=0 and x<0 and define left, right top bottom, from and back edges accordingly

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/box_ball"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/box_ball/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import meshio
import numpy as np
import sys
import warnings

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg
import parameters.read.mesh as rpam

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

warnings.filterwarnings("ignore")
gmsh.initialize()

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'

mesh_file =output_directory + "mesh.msh"

gmsh.model.add("my model")

channel = gmsh.model.occ.addBox(0, 0, 0, rpam.parameters["L"][0], rpam.parameters["L"][1], rpam.parameters["L"][2])
sphere = gmsh.model.occ.addSphere(rpam.parameters["c_r"][0], rpam.parameters["c_r"][1], rpam.parameters["c_r"][2], rpam.parameters["r"])
fluid = gmsh.model.occ.cut([(3, channel)], [(3, sphere)])

gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)

assert volumes == fluid[0]
# these is is the subdomain_id with which the volume [box-sphere] will be read in read_3dmesh_box_ball.py
msh.tag_physical_object(volumes[0], rpam.parameters['volume_id'], gmsh.model, "volume")

surfaces = gmsh.model.occ.getEntities(dim=2)

obstacles = []

# loop through all surfaces and tag them
for surface in surfaces:
    # compute the center of mass of each surface, and recognize according to the coordinates of the center of mass
    center_of_mass = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])

    if np.isclose(center_of_mass[0], 0):
        # the x coordinate of the center of mass is close to  0 -> I am on boundary_l
        msh.tag_physical_object(surface, rpam.parameters['boundary_le_id'], gmsh.model, "boundary_le")

    if np.isclose(center_of_mass[0], rpam.parameters["L"][0]):

        msh.tag_physical_object(surface, rpam.parameters['boundary_ri_id'], gmsh.model, "boundary_ri")

    if np.isclose(center_of_mass[1], 0):

        msh.tag_physical_object(surface, rpam.parameters['boundary_bo_id'], gmsh.model, "boundary_bo")

    if np.isclose(center_of_mass[1], rpam.parameters["L"][1]):

        msh.tag_physical_object(surface, rpam.parameters['boundary_to_id'], gmsh.model, "boundary_to")

    if np.isclose(center_of_mass[2], 0):

        msh.tag_physical_object(surface, rpam.parameters['boundary_ba_id'], gmsh.model, "boundary_ba")

    if np.isclose(center_of_mass[2], rpam.parameters["L"][2]):

        msh.tag_physical_object(surface, rpam.parameters['boundary_fr_id'], gmsh.model, "boundary_fr")

    if (np.allclose(center_of_mass, rpam.parameters["c_r"])):
        # the center of mass is rpam.parameters["c_r"] -> the surface under consideration is the sphere
        obstacles.append(surface[1])  # Save the tag of the sphere surface

        msh.tag_physical_object(surface, rpam.parameters['boundary_sphere_id'], gmsh.model, "boundary_sphere")

# set the resolution close to the obstacle
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", rpam.parameters["resolution_min"])
gmsh.model.mesh.field.setNumber(threshold, "LcMax", rpam.parameters["resolution_max"])
gmsh.model.mesh.field.setNumber(threshold, "DistMin", rpam.parameters["distance_min"])
gmsh.model.mesh.field.setNumber(threshold, "DistMax", rpam.parameters["distance_max"])

gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)

msh.full_write(mesh_file, ['tetra', 'triangle'], metadata, output_directory, False)


# msh.print_mesh_edges_to_csv(mesh_file, output_directory + 'edges.csv')
#
# # create a tetrahedron mesh in which the solid objects (volumes) will be stored
# tetra_mesh = msh.create_mesh(mesh_from_file, "tetra", False)
# meshio.write(output_directory + "tetra_mesh.xdmf", tetra_mesh)
#
# # create a triangle mesh in which the surfaces will be stored
# triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=False)
# meshio.write(output_directory + "triangle_mesh.xdmf", triangle_mesh)
#
# # print the mesh vertices to file
# mesh = msh.read_mesh(output_directory + "tetra_mesh.xdmf")
# io.print_mesh_vertices_to_csv(mesh, output_directory + "vertices.csv")

