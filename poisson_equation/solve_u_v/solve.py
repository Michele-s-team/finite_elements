'''
This code solves the Poisson equation  Nabla u = f expressed in terms of the function u and v_i = \partial_i u

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u_v/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u_v/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u_v/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk_robin $MESH_PATH $SOLUTION_PATH

'''

from fenics import *
import importlib

import sys
import ufl

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import switch_problem as swi

i, j, k, l = ufl.indices(4)

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

J = derivative(vp.F, fsp.psi, fsp.J_uv)
problem = NonlinearVariationalProblem(vp.F, fsp.psi, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)
# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-12,
                  'relative_tolerance': 1e-12,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update(params)
solver.solve()

prout_bc = importlib.import_module(swi.prout_bc)
import print_out_solution
