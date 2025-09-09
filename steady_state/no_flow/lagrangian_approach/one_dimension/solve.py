'''
This file solves for the steady state of a one-dimensional fluid with no flows with the Lagrangian approach

Run with
    python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/lagrangian_approach/one_dimension/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line $MESH_PATH $SOLUTION_PATH;
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
J = derivative(vp.F, fsp.phi, fsp.J_phi)
problem = NonlinearVariationalProblem(vp.F, fsp.phi, vp.bcs, J)
solver = NonlinearVariationalSolver(problem)

'''
# to solve with SNES
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'mumps',
        'line_search': 'bt',  # backtracking line search
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 1000000,
        'report': True,
    }
}

'''

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
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update(params)

solver.solve()

'''
# test n
import command as cmd
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import solution_paths as solpath

cmd.set_gauge('arc_length')

Q_n = VectorFunctionSpace(lmsh.mesh, 'P', 2, dim=2)
n = Function(Q_n)
n.assign(project(geo.normal(fsp.psi), Q_n))

io.full_print(n, 'n', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path, lmsh.mesh,
              'vector')
'''

prout_bc = importlib.import_module(swi.prout_bc)
