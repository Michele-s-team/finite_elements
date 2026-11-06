'''
This code solves the integro-differential Poisson equation 

    Nabla u = s * f  in \Omega
    s = \int_{\partial Omega} u ds
    
on a discontinuous Galerkin function space

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/scalar/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH

'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import switch_problem as swi
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J)

prout_bc = importlib.import_module(swi.prout_bc)
