'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u on a square with no circle inside, and periodic boundary conditions between the left and right edge of the square

clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/periodic/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 solve.py line $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/symmetric/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/periodic/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH;

'''

from fenics import *
import importlib
import sys

import switch_problem as swi
import variational_problem.utils as var_pr

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }

# solve original problem
var_pr.solve_vp(vp.F, fsp.u, vp.bcs, fsp.J_u, parameters=params)

# solve post-processing problem
var_pr.solve_vp(vp.F_pp, fsp.hess_u, vp.bcs_pp, fsp.J_hess_u)

prout_bc = importlib.import_module(swi.prout_bc)

import print_out_solution
