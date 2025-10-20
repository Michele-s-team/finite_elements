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


# 
import function as fu
import input_output as io 
import mesh.utils as msh
import numpy as np
import solution_paths as solpath


class f_sub_mesh_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(2.0 * np.pi * x[0]) / (1+ 10 * x[0])

    def value_shape(self):
        return (1,)

class v_sub_mesh_Expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(2.0 * np.pi * x[0]) / (1+ 10 * x[0])
        values[1] = np.sin(2.0 * np.pi * x[0]) / (1+ 3*x[0]**2)

    def value_shape(self):
        return (2,)
    
    


fsp.f_sub_mesh.interpolate(f_sub_mesh_Expression(element=fsp.Q_sub_mesh.ufl_element()))
fsp.v_sub_mesh.interpolate(v_sub_mesh_Expression(element=fsp.V_sub_mesh.ufl_element()))

fu.transfer_sub_mesh_to_mesh(fsp.f_sub_mesh, fsp.f_mesh)
fu.transfer_sub_mesh_to_mesh(fsp.v_sub_mesh, fsp.v_mesh)

io.full_print(fsp.f_mesh, f'f_mesh', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path,
                  rmsh.lmsh.sub_meshes[0], 'scalar')
io.full_print(fsp.v_mesh, f'v_mesh', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path,
                  rmsh.lmsh.sub_meshes[0], 'vector')


# 

'''
J = [None] * len(rmsh.lmsh.sub_meshes)
problem = [None] * len(rmsh.lmsh.sub_meshes)
solver = [None] * len(rmsh.lmsh.sub_meshes)

# solve problem on sub_mesh[1]
vp[1] = importlib.import_module(swi.vp_sub_mesh_1)

J[1] = derivative(vp[1].F, fsp.u[1], fsp.J_u[1])
problem[1] = NonlinearVariationalProblem(vp[1].F, fsp.u[1], vp[1].bcs, J[1])
solver[1] = NonlinearVariationalSolver(problem[1])
solver[1].parameters.update(params)

solver[1].solve()


# solve problem on sub_mesh[0] by using the solution above on sub_mesh[1] as a BC
vp[0] = importlib.import_module(swi.vp_sub_mesh_0)

J[0] = derivative(vp[0].F, fsp.u[0], fsp.J_u[0])
problem[0] = NonlinearVariationalProblem(vp[0].F, fsp.u[0], vp[0].bcs, J[0])
solver[0] = NonlinearVariationalSolver(problem[0])
solver[0].parameters.update(params)

solver[0].solve()


prout_bc = importlib.import_module(swi.prout_bc)
'''
