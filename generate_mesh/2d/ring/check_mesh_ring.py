'''
This code reads the mesh generated from generate_ring_mesh.py and it runs some checks on it

Run with
    clear; clear; python3 check_mesh_ring.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_ring.py solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import check_mesh_tags_ring
