'''
This code generates a 3d mesh given by a F14 tomcat

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/shapes/tomcat"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/shapes/tomcat/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import os
import sys
import trimesh

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg


# geometry = pygmsh.occ.Geometry()
# model = geometry.__enter__()

metadata = dict()


stl_file = os.path.join('/home/fenics/shared/generate_mesh/3d/shapes/tomcat/input', "mesh.stl")
# msh_file = os.path.join(rarg.args.output_directory, "mesh.msh")

# msh_file = meshio.read(os.path.join('/home/fenics/shared/generate_mesh/3d/shapes/tomcat/input', "mesh.stl"))
'''
meshio.write(mesh_file, stl_file) 

# this is a mesh given by a two-dimensional manifold -> print its vertices, edges and triangles 
msh.print_mesh_vertices_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "vertices.csv"))
msh.print_mesh_edges_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "edges.csv"))
msh.print_mesh_triangles_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "triangles.csv"))
'''


trimesh.creation.capsule(height=2.0, radius=0.5, count=[5, 5]).export(stl_file)


msh.generate_3d_shape_mesh(stl_file, rarg.args.output_directory)

msh.clear_gmsh()
