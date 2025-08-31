'''
This code reads the 3d mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh

Run with
    clear; clear; python3 check_mesh.py [path where to find the mesh] [path where to write the result of the check]
Example:
    clear; clear; MESH_PATH="/home/fenics/shared/generate_mesh/3d/box_ball/solution"; CHECK_PATH="/home/fenics/shared/generate_mesh/3d/box_ball/check"; rm -rf $CHECK_PATH; mkdir $CHECK_PATH; python3 check_mesh.py $MESH_PATH $CHECK_PATH
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.check_tags.box_ball