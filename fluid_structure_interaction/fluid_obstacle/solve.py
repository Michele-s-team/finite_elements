'''
This code solves the dynamics of a fluid in a box (A) with a fluid obstacle (B) in the box. 

The problem has three meshes:
- mesh[0]: a 2d mesh given by the box, including the disk in it. This is divided into 
    * sub_mesh[0]: the disk
    * sub_mesh[1]: the surface between the disk boundary and the box. 
- mesh[1]: a 1d mesh given by a line (the boundary of the circular obstacle laid flat on a line)

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/disk_line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/fluid_obstacle/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_disk_line_a $MESH_PATH $SOLUTION_PATH
 '''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import solution_paths as solpath
import switch_problem as swi

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
J, problem, solver, vp = [[None]*2, None], [[None]*2, None], [[None]*2, None], [[None]*2, None]

# solve the variational problem in sub_mesh[0][1], and obtain the solution 
print('Solving the problem in sub_mesh[0][1]...')
vp[0][1] = importlib.import_module(swi.vp_sub_mesh_0_1)
J[0][1] = derivative(vp[0][1].F, fsp.u[0][1], fsp.J_u[0][1])
problem[0][1] = NonlinearVariationalProblem(vp[0][1].F, fsp.u[0][1], vp[0][1].bcs, J[0][1])
solver[0][1] = NonlinearVariationalSolver(problem[0][1])

solver[0][1].solve()
print('...done.')

print(f'Transferring solution on sub_mesh[0][1] to mesh[1] ...')
msh.transfer_circle_to_line(fsp.u[0][1], fsp.u_0_1_on_1, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])
print(f'... done.')


io.full_print(fsp.u_0_1_on_1, f'u_0_1_on_1', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path,
                  rmsh.lmsh.mesh[1], 'scalar')



# solve the variational problem on mesh[1]
print('Solving the problem in mesh[1]...')
# use the solution obtained for sub_mesh[0][1] in the variational problem on mesh[1]
vp[1] = importlib.import_module(swi.vp_mesh_1)
J[1] = derivative(vp[1].F, fsp.u[1], fsp.J_u[1])
problem[1] = NonlinearVariationalProblem(vp[1].F, fsp.u[1], vp[1].bcs, J[1])
solver[1] = NonlinearVariationalSolver(problem[1])

solver[1].solve()
print('...done.')

print(f'Transferring solution on mesh[1] to sub_mesh[0][0] ...')
msh.transfer_line_to_circle(fsp.u[1], fsp.u_1_on_0_0, rmsh.lmsh.mesh_parameters[0]['c_r'], rmsh.lmsh.mesh_parameters[0]['r'], rmsh.lmsh.mesh_parameters[0]['N'])
print(f'... done.')


vp[0][0] = importlib.import_module(swi.vp_sub_mesh_0_0)
J[0][0] = derivative(vp[0][0].F, fsp.u[0][0], fsp.J_u[0][0])
problem[0][0] = NonlinearVariationalProblem(vp[0][0].F, fsp.u[0][0], vp[0][0].bcs, J[0][0])
solver[0][0] = NonlinearVariationalSolver(problem[0][0])

print('Solving the problem in sub_mesh[0][0]...')
solver[0][0].solve()
print('...done.')


prout_bc = importlib.import_module(swi.prout_bc)
'''