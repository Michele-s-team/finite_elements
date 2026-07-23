'''
This file solves for the steady state of a one-dimensional fluid with no flows with the Lagrangian approach

Run with
    python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]

Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/lagrangian_approach/one_dimension/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_fixed_nu $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/lagrangian_approach/one_dimension/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_solve_nu $MESH_PATH $SOLUTION_PATH;
    
    

the fields in this problem are
psi = psi_{Lagrangian approach}
mu = H
X[alpha] = {X^alpha}_{Lagrangian approach}
u[alpha] = {X^alpha}_{Lagrangian approach} - X_r^alpha
X_ref^alpha is the manifold in the reference configuration 
nu = nu_{Lagrangian approach}
sigma = sigma_{Lagrangian approach}

'''

import colorama as col
from fenics import *
import dolfin
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
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


# to solve with SNES
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'superlu',
        'line_search': 'bt',  # backtracking line search
        'absolute_tolerance': 1e-8,
        'relative_tolerance': 1e-8,
        'maximum_iterations': 1000000,
        'report': True,
    }
}

# Option 1: Use trust region instead of line search
PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')  # Trust region method
PETScOptions.set('snes_max_it', 10000)
PETScOptions.set('snes_monitor')

# Option 2: Use different line search with more aggressive settings
PETScOptions.set('snes_type', 'newtonls')
PETScOptions.set('snes_linesearch_type', 'basic')  # Simple line search
PETScOptions.set('snes_linesearch_damping', 1.0)   # No damping initially
PETScOptions.set('snes_linesearch_max_it', 50)     # More line search iterations

# Option 3: Disable line search completely (use full Newton steps)
PETScOptions.set('snes_linesearch_type', 'basic')
PETScOptions.set('snes_linesearch_damping', 1.0)
PETScOptions.set('snes_linesearch_max_it', 1)
PETScOptions.set('snes_atol', 1e-6)      # Absolute tolerance (much smaller)
PETScOptions.set('snes_rtol', 1e-6)      # Relative tolerance (much smaller) 
PETScOptions.set('snes_stol', 1e-6)      # Step tolerance

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
                  'relaxation_parameter': 0.5,
              }
          }
'''
solver.parameters.update(params)

solver.solve()



prout_bc = importlib.import_module(swi.prout_bc)
