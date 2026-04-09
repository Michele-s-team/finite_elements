'''
This code solves the Poisson equation in mesh[0] which is composed of two sub_meshes, sub_mesh[0][0] and sub_mesh[0][1], which share one boundary. The problem is solved with the monolithic approach, by defining a field u which lives on the full mesh[0]. 

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line $MESH_PATH $SOLUTION_PATH
 '''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)

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




print('Solving the monolithic problem in mesh[0]...')

vp = importlib.import_module(swi.vp)
var_pr.solve_vp(vp.F, fsp.u, vp.bcs, fsp.J_u)

print('...done.')

prout_bc = importlib.import_module(swi.prout_bc)
