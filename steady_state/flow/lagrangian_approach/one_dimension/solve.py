'''
This file solves for the steady state of a one-dimensional fluid with flows with the Lagrangian approach.

Run with
    python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/lagrangian_approach/one_dimension/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_fixed_nu $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/lagrangian_approach/one_dimension/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_solve_nu $MESH_PATH $SOLUTION_PATH;



the fields in this problem are
v[i] = v^i_{Lagrangian approach}
w = w_{Lagrangian approach}
sigma = \sigma_{Lagrangian approach}
psi = psi_{Lagrangian approach}
mu = H
X[i] = {X^i}_{Lagrangian approach}
X_ref^alpha is the manifold in the reference configuration 


'''

import colorama as col
from fenics import *
import dolfin
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

set_log_level(20)
dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']

print("Input diredtory = ", rarg.args.input_directory)
print("Output diredtory = ", rarg.args.output_directory)
print(f"Radius of mesh cell = {col.Fore.BLUE}{rmsh.r_mesh}{col.Style.RESET_ALL}")

# to solve with SNES
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'superlu',
        'line_search': 'bt',  # backtracking line search
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 1000000,
        'report': True,
    }
}

PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')
PETScOptions.set('snes_atol', 1e-12)     # Stricter absolute tolerance
PETScOptions.set('snes_rtol', 1e-12)     # Stricter relative tolerance
PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')

# solve the variational problem
var_pr.solve_vp(vp.F, fsp.phi, vp.bcs, fsp.J_phi, parameters=params)

prout_bc = importlib.import_module(swi.prout_bc)
