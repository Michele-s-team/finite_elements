'''
This code checks the mesh generated from generate_ring_mesh.py

run with
    clear; clear; python3 check_mesh_ring_with_circle.py [path where to find the mesh]
example:
    clear; clear; python3 check_mesh_ring_with_circle.py solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)



import check_mesh_tags_ring_with_circle
