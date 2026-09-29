'''
This code generates a 3d mesh given by a F14 tomcat

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/shapes/tomcat"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/shapes/tomcat/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
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


metadata = dict()


stl_file = os.path.join(rarg.args.output_directory, 'mesh.stl')
# msh_file = os.path.join(rarg.args.output_directory, "mesh.msh")

# msh_file = meshio.read(os.path.join('/home/fenics/shared/generate_mesh/3d/shapes/tomcat/input', "mesh.stl"))
'''
meshio.write(mesh_file, stl_file) 

# this is a mesh given by a two-dimensional manifold -> print its vertices, edges and triangles 
msh.print_mesh_vertices_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "vertices.csv"))
msh.print_mesh_edges_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "edges.csv"))
msh.print_mesh_triangles_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "triangles.csv"))
'''

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


msh.generate_3d_shape_mesh(stl_file, rarg.args.parameter_directory, rarg.args.output_directory)

msh.clear_gmsh()
