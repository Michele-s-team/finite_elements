'''
This code solves the Poisson equation in two sub_meshes, sub_mesh[0][0] and sub_mesh[0][1], which share one boundary. The boundary is laid flat on a line, which is mesh[1]

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/three_domains/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line $MESH_PATH $SOLUTION_PATH
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




'''
here J[i][j] is the Jacobian of the functional for the j-th submesh of the i-th mesh, and similarly for problem, solver, ... 
'''

''''
The three variational problems (VPs) are solved as follows:
1)  Solve Poisson VP on sub_mesh[0][1] for u[0][1] ->
    Obtain 
        a) For test case 1: u[0][1] = (x[0] - cr[0]) + 2 * (x[1] - cr[1])
        b) For test case 2: u[0][1] = 2 * (x[0] - cr[0])**3 + (x[1] - cr[1])**3

2)  Transfer u[0][1] on mesh[1] -> u_0_1_on_1
    r theta = s
    L1 = 2 pi r 
    Given that 
        x[0] = cr[0] + r * cos(s/r), 
        x[1] = cr[1] + r * sin(s/r), 
        where s is the coordinate along mesh[1] and along the circle, we have 

        a) For test case 1: u_0_1_on_1(s)  = (r * cos(s/r)) + 2 * (r * sin(s/r))
        b) For test case 2: u_0_1_on_1(s)  = 2 * (r * cos(s/r))**3 + (r * sin(s/r))**3

3)  Solve on mesh[1] the VP

        u[1]'(s) = u_0_1_on_1(s)

    The solution is 

      a) For test case 1: u[1](s) = C[1] + r^2 (-2 Cos[s/r] + Sin[s/r])
      b) For test case 2: u[1](s) = C[1] + 1/12 r^4 (-9 Cos[s/r] + Cos[(3 s)/r] + 4 (5 + Cos[(2 s)/r]) Sin[s/r])
    
    where I set C[1] -> 0 by adding a Dirichlet BC on the VP on mesh[1]


4)  Transfer u[1](s) to sub_mesh[0][0] and write it in u_1_on_0_0. 
    On the circle 

        a) Test case 1: u_1_on_0_0 = - 2 r * (x[0] - cr[0]) + r * (x[1] - cr[1])
        b) Test case 2: u_1_on_0_0 =  1/12 r (9 r^2 (crx - x) + (-crx + x)^3 + 3 (crx - x) (cry - y)^2 + 
   2 (cry - y)^3 + 18 r^2 (-cry + y) + 6 (crx - x)^2 (-cry + y))

5)  Solve a Poisson problem on sub_mesh[0][0]  with BC on the circle u[0][0] = u_1_on_0_0

    The problem has exact solution 
        a) Test case 1:  u[0][0] = - 2 r * (x[0] - cr[0]) + r * (x[1] - cr[1]),  
        b) Test case 2:  u[0][0] = 1/12 r (9 r^2 (crx - x) + (-crx + x)^3 + 3 (crx - x) (cry - y)^2 + 
   2 (cry - y)^3 + 18 r^2 (-cry + y) + 6 (crx - x)^2 (-cry + y))

    Obtain 
        a) Test case 1): u[0][0] = - 2 r * (x[0] - cr[0]) + r * (x[1] - cr[1]) in sub_mesh[0][0]
        b) Test case 2):  u[0][0] = 1/12 r (9 r^2 (crx - x) + (-crx + x)^3 + 3 (crx - x) (cry - y)^2 + 
   2 (cry - y)^3 + 18 r^2 (-cry + y) + 6 (crx - x)^2 (-cry + y))
'''


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


# 3. solve the variational problem on mesh[1]
# use the solution obtained for sub_mesh[0][1] in the variational problem on mesh[1]

print('Solving the problem in mesh[1]...')

vp[1] = importlib.import_module(swi.vp_mesh_1)
var_pr.solve_vp(vp[1].F, fsp.u[1], vp[1].bcs, fsp.J_u[1])

print('...done.')

# 4. transfer the solution on mesh[1] to sub_mesh[0][0]

print(f'Transferring solution on mesh[1] to sub_mesh[0][0] ...')

msh.transfer_1d_to_2d(fsp.u[1], fsp.u_1_on_0_0, rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'], rmsh.lmsh.parameters['shape_id'])

print(f'... done.')

# 5. solve the problem on sub_mesh[0][0]

print('Solving the problem in sub_mesh[0][0]...')

vp[0][0] = importlib.import_module(swi.vp_sub_mesh_0_0)
var_pr.solve_vp(vp[0][0].F, fsp.u[0][0], vp[0][0].bcs, fsp.J_u[0][0])

print('...done.')


prout_bc = importlib.import_module(swi.prout_bc)
