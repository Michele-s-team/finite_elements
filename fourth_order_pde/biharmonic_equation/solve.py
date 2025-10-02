'''
This code solves the biharmonic equation Nabla Nabla u = f expressed in terms of the function u and v = Nabla u
Run with
    clear; clear; python3 solve.py [problem name] [path where to read the mesh ] [path where to store the solution]
Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/vertex/solution"; SOLUTION_PATH="/home/fenics/shared/fourth_order_pde/biharmonic_equation/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_vertex $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/fourth_order_pde/biharmonic_equation/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring $MESH_PATH $SOLUTION_PATH

'''

from fenics import *
import importlib
import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

J = derivative( vp.F, fsp.psi, fsp.J_Q)
problem = NonlinearVariationalProblem( vp.F, fsp.psi, vp.bcs, J )
solver = NonlinearVariationalSolver( problem )

J_pp = derivative( vp.F_pp, fsp.w, fsp.J_Q_w)
problem_pp = NonlinearVariationalProblem( vp.F_pp, fsp.w, vp.bcs_pp, J_pp )
solver_pp = NonlinearVariationalSolver( problem_pp )


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
PETScOptions.set('snes_max_funcs', 1000000)         # Increase function evaluation limit

solver.parameters.update(params)
solver_pp.parameters.update(params)


solver.solve()
solver_pp.solve()

# u_output, v_output, w_output = psi.split( deepcopy=True )

prout_bc = importlib.import_module(swi.prout_bc)