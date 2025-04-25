'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u
The Hessian of u is solved in a post-processing (pp) variational problem, because one cannot take directly the second derivative of u (u.dx(i).dx(j)) [this would lead to divergences]
run with

clear; clear; python3 solve.py [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
example:
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir -p $SOLUTION_PATH/nodal_values; python3 solve.py /home/fenics/shared/poisson_equation/mesh/solution /home/fenics/shared/poisson_equation/solve_u/$SOLUTION_PATH
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir -p $SOLUTION_PATH/nodal_values; python3 solve.py /home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom/solution /home/fenics/shared/poisson_equation/solve_u/$SOLUTION_PATH
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir -p $SOLUTION_PATH/nodal_values; python3 solve.py /home/fenics/shared/generate_mesh/2d/square/symmetric_top_bottom/mirror_points/solution /home/fenics/shared/poisson_equation/solve_u/$SOLUTION_PATH
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir -p $SOLUTION_PATH/nodal_values; python3 solve.py /home/fenics/shared/generate_mesh/2d/symmetric_ring/solution /home/fenics/shared/poisson_equation/solve_u/$SOLUTION_PATH

change bcs or mesh geometry in line which contain
# CHANGE VARIATIONAL PROBLEM OR MESH HERE
'''

import colorama as col

from fenics import *
from mshr import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import runtime_arguments as rarg

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import read_mesh_ring as rmsh
# import read_mesh_ring_slice as rmsh
# import read_mesh_square_no_circle as rmsh
import read_mesh_square as rmsh

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import variational_problem_bc_ring as vp
# import variational_problem_bc_ring_slice as vp
# import variational_problem_bc_square_no_circle as vp
import variational_problem_bc_square as vp

# n = FacetNormal(mesh)

J = derivative(vp.F, fsp.u, fsp.J_u)
problem = NonlinearVariationalProblem(vp.F, fsp.u, vp.bcs, J)
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

J_pp = derivative(vp.F_pp, fsp.hess_u, fsp.J_hess_u)
problem_pp = NonlinearVariationalProblem(vp.F_pp, fsp.hess_u, [], J_pp)
solver_pp = NonlinearVariationalSolver(problem_pp)

# solve original problem
solver.solve()
# solve pp problem
solver_pp.solve()

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import print_out_bc_ring
# import print_out_bc_ring_slice
# import print_out_bc_square_no_circle
import print_out_bc_square
