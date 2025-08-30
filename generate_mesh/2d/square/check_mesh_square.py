'''
This code checks the mesh generated from generate_mesh.py

run with
    clear; clear; python3 check_mesh_square.py [path where to find the mesh]
example:
    clear; clear; python3 check_mesh_square.py solution
'''

import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.load as lmsh
import mesh.utils as msh
rmsh = importlib.import_module('mesh.read.square')

import mesh.check_tags.square
msh.check_mesh_symmetry(lmsh.mesh, rmsh.parameters["c_r"])

