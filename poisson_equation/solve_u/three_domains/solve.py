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

####################
import mesh.utils as msh

msh.transfer_2d_submesh_to_line(fsp.u[0][0], fsp.u[1], rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'],tol=1e-2)
####################


'''
here J[i][j] is the Jacobian of the functional for the j-th submesh of the i-th mesh, and similarly for problem, solver, ... 
'''
'''
J, problem, solver, vp = [[None]*2, None], [[None]*2, None], [[None]*2, None], [[None]*2, None]

# solve the variational problem in sub_mesh[0][1], and obtain the solution 
vp[0][1] = importlib.import_module(swi.vp_sub_mesh_0_1)
J[0][1] = derivative(vp[0][1].F, fsp.u[0][1], fsp.J_u[0][1])
problem[0][1] = NonlinearVariationalProblem(vp[0][1].F, fsp.u[0][1], vp[0][1].bcs, J[0][1])
solver[0][1] = NonlinearVariationalSolver(problem[0][1])

print('Solving the problem in sub_mesh[0][1]...')
solver[0][1].solve()
print('...done.')

# use the solution obtained for sub_mesh[0][1] to specify the BCs for sub_mesh[0][0], and solve the variational problem in sub_mesh[0][0]
vp[0][0] = importlib.import_module(swi.vp_sub_mesh_0_0)
J[0][0] = derivative(vp[0][0].F, fsp.u[0][0], fsp.J_u[0][0])
problem[0][0] = NonlinearVariationalProblem(vp[0][0].F, fsp.u[0][0], vp[0][0].bcs, J[0][0])
solver[0][0] = NonlinearVariationalSolver(problem[0][0])

print('Solving the problem in sub_mesh[0][0]...')
solver[0][0].solve()
print('...done.')


# solve the variational problem on mesh[1]
vp[1] = importlib.import_module(swi.vp_mesh_1)
J[1] = derivative(vp[1].F, fsp.u[1], fsp.J_u[1])
problem[1] = NonlinearVariationalProblem(vp[1].F, fsp.u[1], vp[1].bcs, J[1])
solver[1] = NonlinearVariationalSolver(problem[1])

print('Solving the problem in mesh[1]...')
solver[1].solve()
print('...done.')


prout_bc = importlib.import_module(swi.prout_bc)
'''