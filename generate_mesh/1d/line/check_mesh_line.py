'''
This code checks the 1d mesh generated from generate_mesh.py
Run with
    clear; clear; python3 check_mesh_line.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_line.py /home/fenics/shared/generate_mesh/1d/line/solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import check_mesh_tags_line
