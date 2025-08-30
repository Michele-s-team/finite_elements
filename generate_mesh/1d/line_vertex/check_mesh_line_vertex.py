'''
This code checks the 1d mesh generated from generate_mesh_line_vertex.py
Run with
    clear; clear; python3 check_mesh_line_vertex.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_line_vertex.py /home/fenics/shared/generate_mesh/1d/line_vertex/solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import mesh.check_tags.line_vertex
