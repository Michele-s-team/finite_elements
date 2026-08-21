'''
This code generates a 3d mesh given by a F14 tomcat

Run it with
    python3 generate_mesh.py [path where to read parameters] [output directory]
Example:
    clear; clear; PARAMETERS_PATH="/home/fenics/shared/generate_mesh/3d/shapes/tomcat"; SOLUTION_PATH="/home/fenics/shared/generate_mesh/3d/shapes/tomcat/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_mesh.py $PARAMETERS_PATH $SOLUTION_PATH
'''

import gmsh
import meshio
import os
import pygmsh
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import runtime_arguments_generate_mesh as rarg


geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

metadata = dict()


mesh_file = os.path.join(rarg.args.output_directory, "mesh.msh")

mesh = meshio.read(os.path.join('/home/fenics/shared/generate_mesh/3d/shapes/tomcat/input', "mesh.stl"))
meshio.write(os.path.join(rarg.args.output_directory, "mesh.msh"), mesh) 

msh.print_mesh_vertices_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "vertices.csv"))
msh.print_mesh_triangles_to_csv(mesh_file, os.path.join(rarg.args.output_directory, "triangles.csv"))

msh.clear_gmsh()
