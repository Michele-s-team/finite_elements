'''
This code generates a 3d mesh given by a box with a surface in it

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/box/surface"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/box/surface/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import numpy as np
import os
import pymeshlab
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

# 1. generate the stl file with the surface



#1.4 generate the surface
'''
# 1.4.a generate a capsule
trimesh.creation.capsule(
        height=rpam.parameters['surface_height'], 
        radius=rpam.parameters['surface_radius'], 
        count=rpam.parameters['surface_n_sections'], 
        transform=transform
    ).export(surface_file)
'''

'''
# 1.4.b generate a torus
# closed circular profile in the (r, z) half-plane: last point equals the first
tab_theta = np.linspace(0.0, 2.0 * np.pi, rpam.parameters['n_sections_min'] + 1)
# generate a circle
cross_sectional_profile = np.column_stack([rpam.parameters['r_max'] + rpam.parameters['r_min'] * np.cos(tab_theta), rpam.parameters['r_min'] * np.sin(tab_theta)])
# revolve the circle about the z axis and obtain a torus
m = trimesh.creation.revolve(cross_sectional_profile, sections=rpam.parameters['n_sections_maj'])
'''

'''
#1.4.c generate a sphere
m = trimesh.creation.icosphere(
    subdivisions=rpam.parameters['surface_subdivisions'],
    radius=rpam.parameters['surface_radius'])
'''

# 1.4.d generate a mesh from an .job file

# 1.4.d.1 load the file
m = trimesh.load('mesh.obj', force='mesh')

#1.4.d.2 isotropic remeshing: replaces the surface triangulation with near-equilateral triangles of uniform size
mesh_set = pymeshlab.MeshSet()
mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=m.vertices, face_matrix=m.faces))
# here `mesh_set` is remeshed by building triangles of relative size `triangle_relative_target_length` with respect to the size of the box enclosing `mesh_set`
mesh_set.remeshing_isotropic_explicit_remeshing(
    targetlen=pymeshlab.Percentage(100.0 * rpam.parameters['triangle_relative_target_length']),
)
current_mesh = mesh_set.current_mesh()
m = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

print(f'bounding box of m: min = {m.bounds[0]}, max = {m.bounds[1]}')

# coordinates of the center of mass of the bounding box of `m`
c = [(m.bounds[1][i]+m.bounds[0][i])/2 for i in range(len(m.bounds[0]))]


#1.1 the generated surface will be rotated according to the rotation matrix `R`
R = tr.rotation_matrix(rpam.parameters['surface_rotation_angle'], rpam.parameters['surface_rotation_axis'])

#1.2 the generated surface will be translated according to the translation matrix `T`: translate `m` in such a way that its bounding box has center of mass equal to surface_translation_vector
T = tr.translation_matrix(
                np.add(np.subtract(np.add(np.multiply(rpam.parameters['L'], 0.5), rpam.parameters['r']), c), rpam.parameters['surface_translation_vector'])
        )



#1.3 compose `T` . `R`
transform = tr.concatenate_matrices(T, R)


#2.  apply the translation + rotation to the generated shape 
m.apply_transform(transform)


m.export(surface_file)

#3. incorporate the surface in the stl file into a box and write this into a 3d mesh
msh.generate_box_surface_mesh(surface_file, rarg.args.parameter_directory, rarg.args.output_directory)