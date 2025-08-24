'''
This code solves the Poisson equation Nabla (u + v) = f with a constraint
clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/constraint/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_constraint_u_v $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/constraint/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_constraint_u2_v2 $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/constraint/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_constraint_grad_u_grad_v $MESH_PATH $SOLUTION_PATH

'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

J = derivative(vp.F, fsp.psi, fsp.J_psi)
problem = NonlinearVariationalProblem(vp.F, fsp.psi, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)

# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  # 'linear_solver': 'superlu',
                  'linear_solver': 'mumps',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.5,
              }
          }
solver.parameters.update(params)

# solve original problem
solver.solve()

prout_bc = importlib.import_module(swi.prout_bc)
