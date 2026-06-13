'''
This file solves for the steady state of a two-dimensional fluid with no flows

This file needs the mesh files, which can be generated with modules in /home/fenics/shared/generate/mesh

Run with
    python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]
    SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a /home/fenics/shared/steady_state/no_flow/mesh/solution /home/fenics/shared/steady_state/no_flow/$SOLUTION_PATH
    rm -r solution; python3 solve.py square_a /home/fenics/shared/steady_state/no_flow/mesh /home/fenics/shared/steady_state/no_flow/solution
    rm -r solution; mpirun -np 6 python3 solve.py square_a /home/fenics/shared/steady_state/no_flow/mesh /home/fenics/shared/steady_state/no_flow/solution

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/symmetric/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_a $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/symmetric/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_a $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_b $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH;
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/steady_state/no_flow/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH;

'''
import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import switch_problem as swi
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

set_log_level(20)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                'linear_solver': 'superlu',
                'absolute_tolerance': 1e-6,
                'relative_tolerance': 1e-6,
                'maximum_iterations': 1000000,
                'relaxation_parameter': 0.95,           
                'preconditioner': 'default'
              }
          }


# J = derivative(vp.F, fsp.psi, fsp.J_psi)
# problem = NonlinearVariationalProblem(vp.F, fsp.psi, vp.bcs, J)
# solver = NonlinearVariationalSolver(problem)
# solver.parameters.update(params)
# solver.solve()

# solve the variational problem
var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi, parameters=params)

# J_pp_tau = derivative(vp.vp_pp.F_pp_tau, fsp.tau, fsp.J_pp_tau)
# problem_pp_tau = NonlinearVariationalProblem(vp.vp_pp.F_pp_tau, fsp.tau, vp.vp_pp.bc_pp_tau, J_pp_tau)
# solver_pp_tau = NonlinearVariationalSolver(problem_pp_tau)
# solver_pp_tau.solve()

# the post-processing ('pp') variational problem used to compute tau
var_pr.solve_vp(vp.vp_pp.F_pp_tau, fsp.tau, vp.vp_pp.bc_pp_tau, fsp.J_pp_tau)

prout_bc = importlib.import_module(swi.prout_bc)

# import print_out_error