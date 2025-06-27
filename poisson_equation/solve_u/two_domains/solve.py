'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u
The Hessian of u is solved in a post-processing (pp) variational problem, because one cannot take directly the second derivative of u (u.dx(i).dx(j)) [this would lead to divergences]

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/square/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_square $MESH_PATH $SOLUTION_PATH
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

J, problem, solver = [], [], []
for i in range(len(rmsh.lmsh.sub_meshes)):

    print(f'* Solving problem {i}...')
    J.append(derivative(vp.F[i], fsp.u[i], fsp.J_u[i]))
    problem.append(NonlinearVariationalProblem(vp.F[i], fsp.u[i], vp.bcs[i], J[i]))
    solver.append(NonlinearVariationalSolver(problem[i]))
    solver[-1].parameters.update(params)

    solver[i].solve()

    print('... done.')

'''

# J_pp = derivative(vp.F_pp, fsp.hess_u, fsp.J_hess_u)
# problem_pp = NonlinearVariationalProblem(vp.F_pp, fsp.hess_u, [], J_pp)
# solver_pp = NonlinearVariationalSolver(problem_pp)

'''
'''
solver_pp.solve()
'''
prout_bc = importlib.import_module(swi.prout_bc)
