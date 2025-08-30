'''
This code checks the mesh generated from generate_mesh.py

run with
    python3 check_mesh_square_ellipse.py [path where to find the mesh]
example:
   python3 check_mesh_square_ellipse.py solution
'''

import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import load_mesh as lmsh
import mesh as msh
import mesh.read.square_ellipse as rmsh

import check_mesh_tags_square_ellipse

msh.check_mesh_symmetry(lmsh.mesh, rmsh.parameters["c"][:2])
