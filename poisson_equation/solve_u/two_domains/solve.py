'''
This code solves the Poisson equation in two sub_meshes, sub_mesh[0] and sub_mesh[1], which share one boundary
The problem is first solved in sub_mesh[1], and the solution u[1] is then used to specify the BCs of the problem of sub_mesh[0]

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/square/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_square $MESH_PATH $SOLUTION_PATH
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_ellipse_circle $MESH_PATH $SOLUTION_PATH
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_line $MESH_PATH $SOLUTION_PATH
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

J = [None] * len(rmsh.lmsh.sub_meshes)
problem = [None] * len(rmsh.lmsh.sub_meshes)
solver = [None] * len(rmsh.lmsh.sub_meshes)

# solve problem 1: the BCs have been already set in vp

J[1] = derivative(vp.F[1], fsp.u[1], fsp.J_u[1])
problem[1] = NonlinearVariationalProblem(vp.F[1], fsp.u[1], vp.bcs[1], J[1])
solver[1] = NonlinearVariationalSolver(problem[1])
solver[1].parameters.update(params)


solver[1].solve()

'''

# solve problem 0 by using the solution of problem 1 to specify the BCs

# set the BC at the interface between sub_mesh[0] and sub_mesh[1] according to the solution fsp,u[1] obtained above
# project fsp.u[1] on fsp.Q[0] and write the result in fsp.u_1_on_0
fsp.u_1_on_0.assign(project((fsp.u[1])**2, fsp.Q[0]))
# impose the BCs for problem on sub_mesh[0], on the ellipse boundary of sub_mesh[0], in terms of fsp.u_1_on_0, and solve problem on sub_mesh[0]
# force reload vp to update bc[0], because u_1_on_0 has changed
importlib.reload(vp)


J[0] = derivative(vp.F[0], fsp.u[0], fsp.J_u[0])
problem[0] = NonlinearVariationalProblem(vp.F[0], fsp.u[0], vp.bcs[0], J[0])
solver[0] = NonlinearVariationalSolver(problem[0])
solver[0].parameters.update(params)

solver[0].solve()
'''
prout_bc = importlib.import_module(swi.prout_bc)
