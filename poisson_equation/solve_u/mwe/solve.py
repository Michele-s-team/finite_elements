'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u
The Hessian of u is solved in a post-processing (pp) variational problem, because one cannot take directly the second derivative of u (u.dx(i).dx(j)) [this would lead to divergences]
run with

clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/vertex/solution"; SOLUTION_PATH="/home/fenics/shared/poisson_equation/solve_u/mwe/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_vertex $MESH_PATH $SOLUTION_PATH
'''

from fenics import *
import sys
import numpy as np



# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh.utils as msh


# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

mesh, sf = msh.read_from_file('/home/fenics/shared/generate_mesh/1d/line/vertex/solution', 'h5')



Q = FunctionSpace(mesh, 'P', 4)

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)
J_u = TrialFunction(Q)
u_exact = Function(Q)



class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = np.cos(2 * np.pi * x[0]) 

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = - 2 * np.pi / 1 * np.sin(2 * np.pi * x[0]) 

    def value_shape(self):
        return (1,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = -( 2 * np.pi )**2 * np.cos(2 * np.pi * x[0]) 
          
    def value_shape(self):
        return (1,)


u_exact.interpolate(u_exact_expression(element=Q.ufl_element()))
f.interpolate(laplacian_u_expression(element=Q.ufl_element()))


'''
bc_u = DirichletBC(Q, u_exact, rmsh.boundary)
bcs = [bc_u]

# variational functional for the original problem (poisson equation)
F = (u.dx(i) * nu_u.dx(i) + f * nu_u) * rmsh.dx \
    - bgeo.facet_normal[i] * (u.dx(i)) * nu_u * rmsh.ds_lr

J = derivative(vp.F, u, J_u)
problem = NonlinearVariationalProblem(vp.F, u, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)

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
solver.parameters.update(params)



# solve original problem
solver.solve()


# solve pp problem
solver_pp.solve()

prout_bc = importlib.import_module(swi.prout_bc)
'''