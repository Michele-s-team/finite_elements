'''
This code checks the mesh generated from generate_mesh.py

Run with
    clear; clear; python3 check_mesh.py [path where to find the mesh] [path where to write the result of the check]
Example:
    clear; clear; MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse/solution"; CHECK_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse/check"; rm -rf $CHECK_PATH; mkdir $CHECK_PATH; python3 check_mesh.py $MESH_PATH $CHECK_PATH
'''

import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.load as lmsh
import mesh.utils as msh
rmsh = importlib.import_module('mesh.read.square_ellipse')

import mesh.check_tags.square_ellipse

msh.check_mesh_symmetry(lmsh.mesh, rmsh.parameters["c"][:2])
