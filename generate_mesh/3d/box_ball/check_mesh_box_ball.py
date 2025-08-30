'''
This code reads the 3d mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh

Run with
    clear; clear; python3 check_mesh_box_ball.py [path where to find the mesh]
Example:
    clear; clear; python3 check_mesh_box_ball.py /home/fenics/shared/generate_mesh/3d/box_ball/solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.check_tags.box_ball