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


# test print to csv file - start
import input_output as io
import os
import solution_paths as solpath
import parameters.read.solution as rpam
import sys

Q = FunctionSpace(rmsh.lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'])
V = VectorFunctionSpace(rmsh.lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'], dim=3)
T = TensorFunctionSpace(rmsh.lmsh.mesh[0], 'DG', rpam.parameters['function_space_degree'], shape=(3,2))


f = Function(Q)
v = Function(V)
t = Function(T)

io.print_to_csvfile(f, os.path.join(solpath.csv_files_path, 'f.csv'), rmsh.lmsh.sf[0])
io.print_to_csvfile(v, os.path.join(solpath.csv_files_path, 'v.csv'), rmsh.lmsh.sf[0])
# io.print_to_csvfile(t, os.path.join(solpath.csv_files_path, 't.csv'), rmsh.lmsh.sf[0])


sys.exit(0)
# test print to csv file - end

var_pr.solve_vp(vp.F, fsp.u, vp.bcs, fsp.J_u, parameters=params)

prout_bc = importlib.import_module(swi.prout_bc)
