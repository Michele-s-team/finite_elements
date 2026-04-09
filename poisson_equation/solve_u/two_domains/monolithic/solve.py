'''
This code solves the Poisson equation in mesh[0] which is composed of two sub_meshes, sub_mesh[0][0] and sub_mesh[0][1], which share one boundary. The problem is solved with the monolithic approach, by defining a field u which lives on the full mesh[0]. 

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line $MESH_PATH $SOLUTION_PATH
 '''

from fenics import *
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)

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




J, problem, solver, vp = [[None]*2, None], [[None]*2, None], [[None]*2, None], [[None]*2, None]

#1. solve the variational problem in sub_mesh[0][1], and obtain the solution 
print('Solving the problem in sub_mesh[0][1]...')

vp[0][1] = importlib.import_module(swi.vp_sub_mesh_0_1)
var_pr.solve_vp(vp[0][1].F, fsp.u[0][1], vp[0][1].bcs, fsp.J_u[0][1])

print('...done.')

# 2. transfer solution on sub_mesh[0][1] to mesh[1]

print(f'Transferring solution on sub_mesh[0][1] to mesh[1] ...')

msh.transfer_2d_to_1d(fsp.u[0][1], fsp.u_0_1_on_1, rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'], rmsh.lmsh.parameters['shape_id'])

io.full_print(fsp.u_0_1_on_1, f'u_0_1_on_1', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)


print(f'... done.')



# 5. solve the problem on sub_mesh[0][0]

print('Solving the problem in sub_mesh[0][0]...')

vp[0][0] = importlib.import_module(swi.vp)
var_pr.solve_vp(vp[0][0].F, fsp.u[0][0], vp[0][0].bcs, fsp.J_u[0][0])

print('...done.')


prout_bc = importlib.import_module(swi.prout_bc)
