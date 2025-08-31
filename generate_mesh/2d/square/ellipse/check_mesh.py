'''
This code checks the mesh generated from generate_mesh.py

run with
    python3 check_mesh.py [path where to find the mesh]
example:
   python3 check_mesh.py solution
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
