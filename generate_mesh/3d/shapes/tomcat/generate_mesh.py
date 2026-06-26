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


# add '/' to output_directory if it is missing
output_directory = io.add_trailing_slash(rarg.args.output_directory)

mesh_file = output_directory + "mesh.msh"



geometry = pygmsh.occ.Geometry()
model = geometry.__enter__()

m = meshio.read(os.path.join('/home/fenics/shared/generate_mesh/3d/shapes/tomcat/input', "mesh.stl"))


model.synchronize()


model.__exit__()
