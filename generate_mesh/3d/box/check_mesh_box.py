'''
This code checks the mesh generated from generate_mesh.py

Run with
    clear; clear; python3 check_mesh_box.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_box.py solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import check_mesh_tags_box
