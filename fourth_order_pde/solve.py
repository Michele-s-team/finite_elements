'''
This code solves the biharmonic equation Nabla Nabla \partial_i (z \partial_i z) = f expressed in terms of the function
- z
- omega[i] = \partial_i z
- mu = \partial_i (z omega_i)
- rho_i = \partial_i mu
- tau = \partial_i rho_i

where the BCs for mu, rho and tau are imposed as Dirichlet BCs with respect to the exact solution, which is known in this case. 

run with
    python3 solve.py [problem name] [path where to read the mesh generated from generate_square_mesh.py or generate_ring_mesh.py] [path where to store the solution]
example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/fourth_order_pde/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_dirichlet $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/fourth_order_pde/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_nitsche $MESH_PATH $SOLUTION_PATH
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

J = derivative(vp.F, fsp.psi, fsp.J_Q)
problem = NonlinearVariationalProblem(vp.F, fsp.psi, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)
# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  # 'linear_solver': 'superlu',
                  'linear_solver': 'mumps',
                  'absolute_tolerance': 1e-12,
                  'relative_tolerance': 1e-12,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update(params)

solver.solve()

prout_bc = importlib.import_module(swi.prout_bc)
