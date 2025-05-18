'''
This code checks the mesh generated from generate_mesh.py

Run with
    clear; clear; python3 check_mesh_square_no_circle.py [path where to find the mesh]
Rxample:
    clear; clear; python3 check_mesh_square_no_circle.py solution
'''

import dolfin
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh as msh
import read_mesh_square_no_circle as rmsh

import check_mesh_tags_square_no_circle
msh.check_mesh_symmetry(lmsh.mesh, [rmsh.L/2, rmsh.h/2])
