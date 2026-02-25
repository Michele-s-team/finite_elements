'''
This file solves for the steady state of a two-dimensional fluid in the presence of tangential flows

This file needs the mesh files, which can be generated with modules in /home/fenics/shared/generate_mesh

Run with
clear; python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]

Example:
    SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a /home/fenics/shared/steady_state/flow/mesh/solution /home/fenics/shared/steady_state/flow/$SOLUTION_PATH
    rm -rf solution; python3 solve.py square_a /home/fenics/shared/steady_state/flow/mesh/solution /home/fenics/shared/steady_state/flow/solution
    rm -rf solution; mpirun -np 6 python3 solve.py square_a /home/fenics/shared/steady_state/flow/mesh/solution /home/fenics/shared/steady_state/flow/solution

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_1 $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/symmetric/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_1 $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_2 $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH;

'''



import colorama as col
import dolfin
from fenics import *
import importlib
import time


import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import input_output as io
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

vp = importlib.import_module(swi.vp)

set_log_level( 20 )
dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

print("Input diredtory = ", rarg.args.input_directory )
print("Output diredtory = ", rarg.args.output_directory )
print(f"Radius of mesh cell = {col.Fore.BLUE}{rmsh.r_mesh:.{io.number_of_decimals}e}{col.Style.RESET_ALL}")



# solve the variational problem
J = derivative( vp.F, fsp.psi, fsp.J_psi )
problem = NonlinearVariationalProblem( vp.F, fsp.psi, vp.bcs, J )
solver = NonlinearVariationalSolver( problem )


#set the solver parameters here
params = {'nonlinear_solver': 'newton',
           'newton_solver':
            {
                # 'linear_solver'           : 'gmres',
                # 'linear_solver'           : 'minres',
                # 'linear_solver'           : 'petsc',
                # 'linear_solver'           : 'richardson',
                # 'linear_solver'           : 'superlu_dist',
                # 'linear_solver'           : 'tfqmr',
                # 'linear_solver'           : 'umfpack',
                # 'linear_solver'           : 'cg',
                # 'linear_solver'           : 'bicgstab',
                'linear_solver'           : 'superlu',
                # 'linear_solver'           : 'mumps',
                # 'linear_solver'           : 'lu',
                'absolute_tolerance'      : 1e-6,
                'relative_tolerance'      : 1e-6,
                'maximum_iterations'      : 1000000,
                'relaxation_parameter'    : 0.95,
             }
}
solver.parameters.update(params)

'''
#set the solver parameters here
params ={"newton_solver": {"linear_solver": 'superlu'}}
solver.parameters.update(params)
'''

#the post-processing ('pp') variational problem used to compute tau
J_pp_tau = derivative( vp.vp_pp.F_pp_tau, fsp.tau, fsp.J_pp_tau )
J_pp_d = derivative( vp.vp_pp.F_pp_d, fsp.d, fsp.J_pp_d )
problem_pp_tau = NonlinearVariationalProblem( vp.vp_pp.F_pp_tau, fsp.tau, [], J_pp_tau )
problem_pp_d = NonlinearVariationalProblem( vp.vp_pp.F_pp_d, fsp.d, [], J_pp_d )
solver_pp_tau = NonlinearVariationalSolver( problem_pp_tau )
solver_pp_d = NonlinearVariationalSolver( problem_pp_d )

start_time = time.time()
solver.solve()
end_time = time.time()

solver_pp_tau.solve()
solver_pp_d.solve()

prout_bc = importlib.import_module(swi.prout_bc)

# import print_out_error

'''
import print_out_time as prt
prt.print_time(end_time - start_time)
'''