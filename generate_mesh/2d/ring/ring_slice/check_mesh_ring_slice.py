'''
This code checks the mesh generated from generate_mesh_ring_slice.py

Run with
    clear; clear; python3 check_mesh_ring_slice.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_ring_slice.py solution
'''
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.check_tags.ring_slice
