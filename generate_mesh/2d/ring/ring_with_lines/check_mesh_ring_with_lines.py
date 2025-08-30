'''
This code checks the mesh generated from generate_ring_mesh.py

run with
    clear; clear; python3 check_mesh_ring_with_lines.py [path where to find the mesh]
example:
    clear; clear; python3 check_mesh_ring_with_lines.py solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh.utils as msh
import mesh.read.ring as rmsh

import check_mesh_tags_ring
msh.check_mesh_symmetry(lmsh.mesh, rmsh.parameters["c_r"])
