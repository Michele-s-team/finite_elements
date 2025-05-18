'''
This code checks the mesh generated from generate_ring_mesh.py

Run with
    clear; clear; python3 check_mesh_disk.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_disk.py solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh as msh
import read_mesh_disk as rmsh


import check_mesh_tags_disk
msh.check_mesh_symmetry(lmsh.mesh, rmsh.c_r)
