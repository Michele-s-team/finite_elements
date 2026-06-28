'''
This code generates a 3d mesh given by a box

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/box"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/box/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import numpy as np
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

print(f'parameter_directory: {rarg.args.parameter_directory}\noutput_directory: {rarg.args.output_directory}')

# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

# write into metadata the file format wich which the mesh will be written
metadata = rpam.parameters.copy()
metadata['file_format'] = 'xdmf'


mesh_file = output_directory + "mesh.msh"

geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()


box = model.add_box([0, 0, 0], rpam.parameters["L"], mesh_size=rpam.parameters["resolution"])

model.synchronize()

# tag 3d objects
volumes = gmsh.model.getEntities(dim=3)
for volume in volumes:
    gmsh.model.addPhysicalGroup(3, [volume[1]], rpam.parameters['volume_id'])  # Tag 1 for volume
    gmsh.model.setPhysicalName(3, rpam.parameters['volume_id'], "volume")

# tag 2d objects
surfaces = gmsh.model.occ.getEntities(dim=2)

for surface in surfaces:
    # compute the center of mass of each surface, and recognize according to the coordinates of the center of mass
    center_of_mass = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])

    if np.isclose(center_of_mass[0], 0):
        # the x coordinate of the center of mass is close to  0 -> I am on boundary_l
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], rpam.parameters['boundary_le_id'])
        gmsh.model.setPhysicalName(surface[0], rpam.parameters['boundary_le_id'], "boundary_le")

    if np.isclose(center_of_mass[0], rpam.parameters["L"][0]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], rpam.parameters['boundary_ri_id'])
        gmsh.model.setPhysicalName(surface[0], rpam.parameters['boundary_ri_id'], "boundary_ri")

    if np.isclose(center_of_mass[1], 0):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], rpam.parameters['boundary_bo_id'])
        gmsh.model.setPhysicalName(surface[0], rpam.parameters['boundary_bo_id'], "boundary_bo")

    if np.isclose(center_of_mass[1], rpam.parameters["L"][1]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], rpam.parameters['boundary_to_id'])
        gmsh.model.setPhysicalName(surface[0], rpam.parameters['boundary_to_id'], "boundary_to")

    if np.isclose(center_of_mass[2], 0):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], rpam.parameters['boundary_ba_id'])
        gmsh.model.setPhysicalName(surface[0], rpam.parameters['boundary_ba_id'], "boundary_ba")

    if np.isclose(center_of_mass[2], rpam.parameters["L"][2]):
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], rpam.parameters['boundary_fr_id'])
        gmsh.model.setPhysicalName(surface[0], rpam.parameters['boundary_fr_id'], "boundary_fr")

geometry.generate_mesh(dim=3)
gmsh.write(mesh_file)

mesh_from_file = meshio.read(mesh_file)
msh.full_write(mesh_file, ['tetra', 'triangle'], metadata, output_directory, False)

model.__exit__()
