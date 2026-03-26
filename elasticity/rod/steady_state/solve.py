'''
This code solves for the deformation field of an elastic rod subjected to gravity at steady state

Run with
    python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/elasticity/rod/steady_state/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_a $MESH_PATH $SOLUTION_PATH
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


var_pr.solve_vp(vp.F, fsp.u, vp.bcs, fsp.J_u, parameters=params)

prout_bc = importlib.import_module(swi.prout_bc)
