'''
This code generates a 3d mesh given by a box with a surface in it

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/box/surface"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/box/surface/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import os
import sys
import trimesh
from trimesh import transformations as tr

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.utils as msh
import parameters.read.mesh as rpam
import runtime_arguments_generate_mesh as rarg


surface_file = os.path.join(rarg.args.output_directory, 'mesh.stl')

'''
# test volume_integral_tetrahedron - start
import calculus as cal
import numpy as np

p_1 = [0.3, 0.2, 0.5]
p_2 = [4,2.354,0]
p_3 = [3.5,2,-1]
p_4 = [2,2.54,-2.43]

tetrahedron = [p_1, p_2, p_3, p_4]

def g(x):
    return x[0]**2 - x[1]* np.cos(2*np.pi*x[2])

print(f'integral = {cal.volume_integral_tetrahedron(g, tetrahedron)}')

sys.exit(1)
# test volume_integral_tetrahedron - end
'''


# 1. generate the stl file with the surface

#1.1 the generated surface will be rotated according to the rotation matrix `R`
R = tr.rotation_matrix(rpam.parameters['surface_rotation_angle'], rpam.parameters['surface_rotation_axis'])

#1.2 the generated surface will be translated according to the translation matrix `T`
T = tr.translation_matrix(rpam.parameters['surface_translation_vector'])

#1.3 compose `T` . `R`
transform = tr.concatenate_matrices(T, R)

'''
#1.4 generate the surface
trimesh.creation.capsule(
        height=rpam.parameters['surface_height'], 
        radius=rpam.parameters['surface_radius'], 
        count=rpam.parameters['surface_n_sections'], 
        transform=transform
    ).export(surface_file)
'''

m = trimesh.creation.icosphere(
    subdivisions=rpam.parameters['surface_subdivisions'],
    radius=rpam.parameters['surface_radius'])

m.apply_transform(transform)
m.export(surface_file)

# 2. incorporate the surface in the stl file into a box and write this into a 3d mesh
msh.generate_box_surface_mesh(surface_file, rarg.args.parameter_directory, rarg.args.output_directory)