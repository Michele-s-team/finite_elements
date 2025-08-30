'''
This code checks the mesh generated from generate_half_circle_with_line_mesh.py

Run with
    clear; clear; python3 check_mesh_half_circle_with_line.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_half_circle_with_line.py solution
'''
from fenics import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.check_tags.half_circle_with_line
