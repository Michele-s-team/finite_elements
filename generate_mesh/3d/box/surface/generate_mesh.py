'''
This code generates a 3d mesh given by a box with a surface in it

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/box/surface"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/box/surface/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import os
import shutil
import sys
import trimesh
from trimesh import transformations as tr

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.utils as msh
import parameters.read.mesh as rpam
import runtime_arguments_generate_mesh as rarg


stl_file = os.path.join(rarg.args.output_directory, 'mesh.stl')

# 1. generate the stl file with the shape
# the generated shape will be rotated according to the rotation matrix `R`
R = tr.rotation_matrix(rpam.parameters['shape_rotation_angle'], rpam.parameters['shape_rotation_axis'])
# the generated shape will be translated according to the translation matrix `T`
T = tr.translation_matrix(rpam.parameters['shape_translation_vector'])
# compose `T` . `R`
transform = tr.concatenate_matrices(T, R)
# generate the shape
trimesh.creation.capsule(
        height=rpam.parameters['shape_height'], 
        radius=rpam.parameters['shape_radius'], 
        count=rpam.parameters['shape_n_sections'], 
        transform=transform
    ).export(stl_file)


# 2. incorporate the shape in the stl file into a box and write this into a 3d mesh
msh.generate_box_surface_mesh(stl_file, rarg.args.parameter_directory, rarg.args.output_directory)

msh.clear_gmsh()
