"""
This code solves for the dynamics of the Navier Stokes equations with a rigid obstacle which can rotate about a fixed point,
 on a flat manifold Crank Nicholson discretization scheme

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/rigid_obstacle/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_ellipse $MESH_PATH $SOLUTION_PATH

Note that all sections of the code which need to be changed when an external parameter (e.g., the inflow velocity, the length of the rectangle, etc...) is changed are bracketed by
#CHANGE PARAMETERS HERE
"""

import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

import print_out_solution as pr_sol

dt = rpam.parameters["T"] / rpam.parameters["num_steps"]  # time step size

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

# initialize values
fsp.theta_n = rpam.parameters["theta_0"]
fsp.omega_n = rpam.parameters["omega_0"]
fsp.theta_n_1 = rpam.parameters["theta_0"]
fsp.omega_n_1 = rpam.parameters["omega_0"]

rmsh = importlib.import_module(swi.rmsh)
ap_ellipse = importlib.import_module(swi.ap_ellipse)
vp_fluid = importlib.import_module(swi.vp_fluid)
vp_mesh = importlib.import_module(swi.vp_mesh)
pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

# set the initial profiles
fsp.v_n_1.interpolate(vp_fluid.v_expression(element=fsp.Q_v.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.sigma_n_12.interpolate(vp_fluid.sigma_expression(element=fsp.Q_phi.ufl_element()))
fsp.sigma_n_32.assign(fsp.sigma_n_12)

print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters["num_steps"]):
    # Update current time
    t += dt
    step += 1

    # step 1): update theta and omega
    print('Solving theta problem ...', flush=True)
    ap_ellipse = importlib.reload(ap_ellipse)

    fsp.theta_n = fsp.theta_n_1 + dt * fsp.omega_n_1
    fsp.omega_n = fsp.omega_n_1 + dt / rpam.parameters["I_ellipse"] * ap_ellipse.M_ellipse
    print('... done.', flush=True)

    # step 2): update u and u_dot (mesh problem)
    print('Solving mesh problem ...', flush=True)

    vp_mesh = importlib.reload(vp_mesh)


    # solve for u and u_dot

    '''
    J_u = derivative(vp_mesh.F_u, fsp.u_n, fsp.J_u)
    problem_u = NonlinearVariationalProblem(vp_mesh.F_u, fsp.u_n, vp_mesh.bcs, J_u)
    solver_u = NonlinearVariationalSolver(problem_u)
    solver_u.parameters.update(params)
    solver_u.solve()
    '''
    var_pr.solve_vp(vp_mesh.F_u, fsp.u_n, vp_mesh.bcs, fsp.J_u, params)

    '''
    J_u_dot = derivative(vp_mesh.F_u_dot, fsp.u_dot_n, fsp.J_u_dot)
    problem_u_dot = NonlinearVariationalProblem(vp_mesh.F_u_dot, fsp.u_dot_n, vp_mesh.bcs_dot, J_u_dot)
    solver_u_dot = NonlinearVariationalSolver(problem_u_dot)
    solver_u_dot.parameters.update(params)
    solver_u_dot.solve()
    '''
    var_pr.solve_vp(vp_mesh.F_u_dot, fsp.u_dot_n, vp_mesh.bcs_dot, fsp.J_u_dot, params)



    print('... done.', flush=True)

    # step 3) update v_n and sigma_n_12 (fluid problem)
    print('Solving fluid problem ...', flush=True)

    vp_fluid = importlib.reload(vp_fluid)

    # step 3.1
    '''
    J_fluid_1 = derivative(vp_fluid.F_v_, fsp.v_, fsp.J_v_)
    problem_fluid_1 = NonlinearVariationalProblem(vp_fluid.F_v_, fsp.v_, vp_fluid.bc_v_, J_fluid_1)
    solver_fluid_1 = NonlinearVariationalSolver(problem_fluid_1)
    solver_fluid_1.solve()
    '''
    var_pr.solve_vp(vp_fluid.F_v_, fsp.v_, vp_fluid.bc_v_, fsp.J_v_)

    # Step 3.2: surface_tension correction step
    '''
    J_fluid_2 = derivative(vp_fluid.F_phi, fsp.phi, fsp.J_phi)
    problem_fluid_2 = NonlinearVariationalProblem(vp_fluid.F_phi, fsp.phi, vp_fluid.bc_phi, J_fluid_2)
    solver_fluid_2 = NonlinearVariationalSolver(problem_fluid_2)
    solver_fluid_2.solve()
    '''
    var_pr.solve_vp(vp_fluid.F_phi, fsp.phi, vp_fluid.bc_phi, fsp.J_phi)

    # step 3.3
    J_fluid_3 = derivative(vp_fluid.F_v_n, fsp.v_n, fsp.J_v_n)
    problem_fluid_3 = NonlinearVariationalProblem(vp_fluid.F_v_n, fsp.v_n, [], J_fluid_3)
    solver_fluid_3 = NonlinearVariationalSolver(problem_fluid_3)
    solver_fluid_3.solve()

    print('... done.', flush=True)

    pr_bc.print_bcs()

    # update the fields
    # 1)
    fsp.theta_n_1 = fsp.theta_n
    fsp.omega_n_1 = fsp.omega_n

    # 2)
    fsp.u_n_2.assign(fsp.u_n_1)
    fsp.u_n_1.assign(fsp.u_n)

    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(fsp.u_dot_n)

    # 3)
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - fsp.phi)

    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(fsp.v_n)

    fsp.sigma_n_32.assign(fsp.sigma_n_12)
    
    if step % rpam.parameters['print_out_stride'] == 0:
        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        pr_sol.print_solution(t, step, dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters["T"])), flush=True)

print("... done.", flush=True)
