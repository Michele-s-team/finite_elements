'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u, on a mesh (mesh[0]) given by a rectangle with a shape in it. Both the rectangle and the shape are meshed inside. 

I use a discontinuous Galerkin function space, by allowing for a jump of grad u across the shape boundary. 

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_a $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_b $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/two_squares_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py two_squares_no_circle_a $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/two_squares_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/two_domains/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py two_squares_no_circle_b $MESH_PATH $SOLUTION_PATH

'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import switch_problem as swi
import variational_problem.utils as var_pr

# test interpolate_dg for vectors  and tensors - start
import sys
import mesh.utils as msh
import parameters.read.solution as rpam
rmsh = importlib.import_module(swi.rmsh)
'''
#1. test for scalar
Q = FunctionSpace(rmsh.lmsh.mesh, 'DG', rpam.parameters['function_space_degree'])
u = Function(Q)

class u_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 

    def value_shape(self):
        return (1,)

msh.interpolate_dg(u, u_expression(), rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])
'''

'''
#2. test for vector
V = VectorFunctionSpace(rmsh.lmsh.mesh, 'DG', rpam.parameters['function_space_degree'])
v = Function(V)

class v_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 
        values[1] = 1 - x[0] ** 2 + 2 * x[1] ** 2 


    def value_shape(self):
        return (2,)

msh.interpolate_dg(v, v_expression(), rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])
'''
# 3. test for tensor
T = TensorFunctionSpace(rmsh.lmsh.mesh, 'DG', rpam.parameters['function_space_degree'], shape=(2, 3))
t = Function(T)

class t_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2 
        values[1] = 1 + x[0] ** 2 + 2 * x[1] ** 2 
        values[2] = 1 + x[0] ** 2 + 2 * x[1] ** 2 
        values[3] = 1 + x[0] ** 2 + 2 * x[1] ** 2 
        values[4] = 1 + x[0] ** 2 + 2 * x[1] ** 2 
        values[5] = 1 - x[0] ** 2 + 2 * x[1] ** 2 


    def value_shape(self):
        return (2, 3)

msh.interpolate_dg(t, t_expression(), rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])



sys.exit(1)
# test interpolate_dg for vectors  and tensors - end


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-10,
                  'relative_tolerance': 1e-10,
                  'maximum_iterations': 1000000,
                #   'relaxation_parameter': 0.95,
              }
          }




var_pr.solve_vp(vp.F, fsp.u, vp.bcs, fsp.J_u, parameters=params)

prout_bc = importlib.import_module(swi.prout_bc)
