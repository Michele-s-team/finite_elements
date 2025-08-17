'''
This file solves for the steady state of a two-dimensional fluid with no flows in the spherically symmetric case,
by allowing for overhangs, following the approach in 'Lagrangian approach' from the paper dereny et al 2002


Run with
    python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/dereny_approach/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH;

'''

import colorama as col
from fenics import *
import dolfin
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

set_log_level(20)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

print("Input diredtory = ", rarg.args.input_directory)
print("Output diredtory = ", rarg.args.output_directory)
print(f"Radius of mesh cell = {col.Fore.BLUE}{rmsh.r_mesh}{col.Style.RESET_ALL}")


# solve the variational problem
J = derivative(vp.F, fsp.phi, fsp.J_psi)
problem = NonlinearVariationalProblem(vp.F, fsp.phi, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)

# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  # 'linear_solver'           : 'mumps',
                  # 'linear_solver':   'lu',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.05,
              }
          }
solver.parameters.update(params)

solver.solve()

'''
prout_bc = importlib.import_module(swi.prout_bc)
'''
