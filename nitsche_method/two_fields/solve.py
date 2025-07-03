'''
This code solves a PDE for two fields u, v  by imposing the BCs with Nitsche's method. Run with

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; SOLUTION_PATH="/home/fenics/shared/nitsche_method/two_fields/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk $MESH_PATH $SOLUTION_PATH
'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

solve(vp.F == 0, fsp.u)

prout_bc = importlib.import_module(swi.prout_bc)
