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
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)
vp = ['','']

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

# solve problem on sub_mesh[1]
vp[1] = importlib.import_module(swi.vp_sub_mesh_1)

var_pr.solve_vp(vp[1].F, fsp.u[1],  vp[1].bcs,  fsp.J_u[1], parameters=params)

# solve problem on sub_mesh[0] by using the solution above on sub_mesh[1] as a BC
vp[0] = importlib.import_module(swi.vp_sub_mesh_0)

var_pr.solve_vp(vp[0].F, fsp.u[0], vp[0].bcs, fsp.J_u[0], parameters=params)


prout_bc = importlib.import_module(swi.prout_bc)
