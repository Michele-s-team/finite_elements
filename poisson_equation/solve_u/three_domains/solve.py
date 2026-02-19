'''
This code solves the Poisson equation in two sub_meshes, sub_mesh[0] and sub_mesh[1], which share one boundary
The problem is first solved in sub_mesh[1], and the solution u[1] is then used to specify the BCs of the problem of sub_mesh[0]

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/disk_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/three_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_disk_line $MESH_PATH $SOLUTION_PATH
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
vp_mesh_0 = ['','']

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


J_mesh_0 = [None] * len(rmsh.lmsh.sub_meshes[0])
problem_mesh_0 = [None] * len(rmsh.lmsh.sub_meshes[0])
solver_mesh_0 = [None] * len(rmsh.lmsh.sub_meshes[0])

# solve problem on sub_mesh[1]
vp_mesh_0[1] = importlib.import_module(swi.vp_sub_mesh_0_1)

J_mesh_0[1] = derivative(vp_mesh_0[1].F, fsp.u[1], fsp.J_u[1])
problem_mesh_0[1] = NonlinearVariationalProblem(vp_mesh_0[1].F, fsp.u[1], vp_mesh_0[1].bcs, J_mesh_0[1])
solver_mesh_0[1] = NonlinearVariationalSolver(problem_mesh_0[1])
solver_mesh_0[1].parameters.update(params)

solver_mesh_0[1].solve()

'''
# solve problem on sub_mesh[0] by using the solution above on sub_mesh[1] as a BC
vp_mesh_0[0] = importlib.import_module(swi.vp_sub_mesh_0_0)

J_mesh_0[0] = derivative(vp_mesh_0[0].F, fsp.u[0], fsp.J_u[0])
problem_mesh_0[0] = NonlinearVariationalProblem(vp_mesh_0[0].F, fsp.u[0], vp_mesh_0[0].bcs, J_mesh_0[0])
solver_mesh_0[0] = NonlinearVariationalSolver(problem_mesh_0[0])
solver_mesh_0[0].parameters.update(params)

solver_mesh_0[0].solve()
'''

prout_bc = importlib.import_module(swi.prout_bc)
