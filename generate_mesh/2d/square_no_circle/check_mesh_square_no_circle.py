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



import check_mesh_tags_square_no_circle

