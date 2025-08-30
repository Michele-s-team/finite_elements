'''
This code checks the mesh generated from generate_mesh.py

Run with
    clear; clear; python3 check_mesh_square_no_circle.py [path where to find the mesh]
Rxample:
    clear; clear; python3 check_mesh_square_no_circle.py solution
'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh.utils as msh
rmsh = importlib.import_module('mesh.read.square_no_circle')

import mesh.check_tags.square_no_circle
msh.check_mesh_symmetry(lmsh.mesh, [rmsh.parameters["L"]/2, rmsh.parameters["h"]/2])
