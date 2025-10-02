'''
This code solves the third-order equation d^3 u /dx^3 = f expressed in terms of the function u and v =  du/dx
Run with
    clear; clear; python3 solve.py [problem name] [path where to read the mesh ] [path where to store the solution]
Example:
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/vertex/solution"; SOLUTION_PATH="/home/fenics/shared/third_order_pde/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_vertex_dirichlet $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/vertex/solution"; SOLUTION_PATH="/home/fenics/shared/third_order_pde/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_vertex_nitsche $MESH_PATH $SOLUTION_PATH

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



'''
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'superlu',
        'line_search': 'bt', 
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 1000000,
        'report': True,
    }
}

PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')
PETScOptions.set('snes_atol', 1e-12)     
PETScOptions.set('snes_rtol', 1e-12)     
PETScOptions.set('snes_stol', 1e-8)      
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 1000000)         

solver.parameters.update(params)
solver_pp.parameters.update(params)
'''


solver.solve()

prout_bc = importlib.import_module(swi.prout_bc)