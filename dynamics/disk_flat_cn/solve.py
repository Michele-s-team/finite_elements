"""
This code solves for the dynamics of the Navier Stokes equations on a flat  manifold (a disk) with Crank Nicholson discretization scheme, with boundary conditions that fix only the traction 

run with:
rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/disk/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/disk_flat_cn/solution"; rm -rf $SOLUTION_PATH; python3 solve.py disk $MESH_PATH $SOLUTION_PATH
"""

import colorama as col
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
import print_out_solution as pr_sol

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)
print(f"Radius of mesh cell = {col.Fore.BLUE}{rmsh.r_mesh}{col.Style.RESET_ALL}")


# set the initial profiles
fsp.v_n_1.interpolate(vp.v0_Expression(element=fsp.Q_v.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.sigma_n_12.interpolate(vp.sigma0_Expression(element=fsp.Q_sigma.ufl_element()))
fsp.sigma_n_32.assign(fsp.sigma_n_12)



print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['num_steps']):
    # Update current time
    t += vp.dt
    step += 1

    vp = importlib.import_module(swi.vp)

    # step 1
    J1 = derivative(vp.F1, fsp.v_, fsp.J_v_)
    problem1 = NonlinearVariationalProblem(vp.F1, fsp.v_, [], J1)
    solver1 = NonlinearVariationalSolver(problem1)
    solver1.solve()

    # Step 2: surface_tension correction step
    J2 = derivative(vp.F2, fsp.phi_omega, fsp.J_phi_omega)
    problem2 = NonlinearVariationalProblem(vp.F2, fsp.phi_omega, [], J2)
    solver2 = NonlinearVariationalSolver(problem2)
    solver2.solve()

    # step 3
    J3 = derivative(vp.F3, fsp.v_n, fsp.J_v_n)
    problem3 = NonlinearVariationalProblem(vp.F3, fsp.v_n, [], J3)
    solver3 = NonlinearVariationalSolver(problem3)
    solver3.solve()

    phi_output, omega_output = fsp.phi_omega.split(deepcopy=True)


    # pr_bc.print_bcs()

    # obtain fsp.sigma_n from fsp.phi by using the definition of fsp.phi
    fsp.sigma_n_12.assign(project(fsp.sigma_n_32 - phi_output, fsp.Q_sigma))

    # Update previous solution
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(fsp.v_n)
    fsp.sigma_n_32.assign(fsp.sigma_n_12)

    pr_sol.print_solution(t, step, vp.dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
