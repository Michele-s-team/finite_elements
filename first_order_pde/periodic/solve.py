'''
This code solves a first-order  equation in the unknown u with periodic boundary conditions 

clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/first_order_pde/periodic/solution"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 solve.py line $MESH_PATH $SOLUTION_PATH;

'''

from fenics import *
import importlib
import runtime_arguments as rarg
import sys
import switch_problem as swi

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


J = derivative(vp.F, fsp.u, fsp.J_u)
problem = NonlinearVariationalProblem(vp.F, fsp.u, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)

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
solver.parameters.update(params)

J_pp = derivative(vp.F_pp, fsp.hess_u, fsp.J_hess_u)
problem_pp = NonlinearVariationalProblem(vp.F_pp, fsp.hess_u, [], J_pp)
solver_pp = NonlinearVariationalSolver(problem_pp)


fsp.u.assign(Constant(1))

# solve original problem
solver.solve()
# solve pp problem
# solver_pp.solve()

prout_bc = importlib.import_module(swi.prout_bc)

import print_out_solution
