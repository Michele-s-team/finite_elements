'''
This code solves for the dynamics of the deformation field of an elastic rod subjected to gravity

Run with
    python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/elasticity/rod/dynamics/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH
'''

from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import read_parameters as rpam
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

# dolfin.parameters["form_compiler"]["quadrature_degree"] = 10


print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

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

# set the initial profiles
fsp.u_n_1.interpolate(vp.u_0_expression(element=fsp.Q.ufl_element()))
fsp.v_n_1.interpolate(vp.u_0_expression(element=fsp.Q.ufl_element()))

print("Starting time iteration ...", flush=True)

# Time-stepping
t = 0
step = 0

for n in range(rpam.parameters['num_steps']):
    # Update current time
    t += vp.dt
    step += 1

    vp = importlib.import_module(swi.vp)

    J = derivative(vp.F, fsp.u_n_1, fsp.J_u)
    problem = NonlinearVariationalProblem(vp.F, fsp.u_n_1, vp.bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters.update(params)
    solver.solve()

    fsp.u_n_1.assign(fsp.u_n)
    fsp.v_n_1.assign(fsp.v_n)

    # pr_sol.print_solution(t, step, vp.dt)

    print("\t%.2f %%" % (100.0 * (t / vp.T)), flush=True)

print("... done.", flush=True)
