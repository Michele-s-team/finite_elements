'''
This code solves for the deformation field of a mesh, obtained from elasticity theory

clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_ellipse $MESH_PATH $SOLUTION_PATH

'''

from fenics import *
import importlib
import dolfin
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
vp_dot = importlib.import_module(swi.vp_dot)

J = derivative(vp.F, fsp.u, fsp.J_u)
problem = NonlinearVariationalProblem(vp.F, fsp.u, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)

J_dot = derivative(vp_dot.F_dot, fsp.u_dot, fsp.J_u_dot)
problem_dot = NonlinearVariationalProblem(vp_dot.F_dot, fsp.u_dot, vp_dot.bcs_dot, J_dot)
solver_dot = NonlinearVariationalSolver(problem_dot)


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
solver_dot.parameters.update(params)


# solve  problem for u
solver.solve()
# solve problem for u_dot
solver_dot.solve()

prout_bc = importlib.import_module(swi.prout_bc)
prout_bc_dot = importlib.import_module(swi.prout_bc_dot)
