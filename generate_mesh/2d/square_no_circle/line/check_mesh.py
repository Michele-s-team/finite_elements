'''
This code checks the mesh generated from generate_mesh.py

run with
    clear; clear; python3 check_mesh.py [path where to find the mesh]
example:
    clear; clear; python3 check_mesh.py solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh.utils as msh
import mesh.read.square_no_circle_line

'''

import mesh.check_tags.square_no_circle_line

'''