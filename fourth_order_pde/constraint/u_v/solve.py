'''
This code solves the fourth-order PDE
 Nabla Nabla \partial_i ((z+u) \partial_i (z+u)) = f with an additional constraint which relates z and u,
 expressed in terms of the function
- z, u
- omega[i] = \partial_i (z+u)
- mu = \partial_i ((z+u) omega_i)

where the BCs for mu, rho and tau are imposed as Dirichlet BCs with respect to the exact solution, which is known in this case. 

Run with
    python3 solve.py [problem name] [path where to read the mesh generated from generate_mesh.pu] [path where to store the solution]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/fourth_order_pde/constraint/u_v/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring $MESH_PATH $SOLUTION_PATH
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
                  'linear_solver': 'superlu',
                  # 'linear_solver': 'mumps',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update(params)

solver.solve()

prout_bc = importlib.import_module(swi.prout_bc)
