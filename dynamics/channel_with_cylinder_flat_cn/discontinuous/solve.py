"""
This code solves for the dynamics of the Navier Stokes equations on a flat  manifold  with Crank Nicholson discretization scheme, by using discontinuous function spaces

run with:
rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_flat_cn/discontinuous/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH
"""

import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi
import print_out_solution as pr_sol

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']



# set the initial profiles
class v_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    
    
class sigma_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0.0

    def value_shape(self):
        return (1,)



msh.interpolate_dg(fsp.v_n_1, v_0_expression(), rmsh.sf)
fsp.v_n_2.assign(fsp.v_n_1)

msh.interpolate_dg(fsp.sigma_n_12, sigma_0_expression(), rmsh.sf)
fsp.sigma_n_32.assign(fsp.sigma_n_12)

# print test - start
import input_output as io
import solution_paths as solpath

io.full_print(fsp.v_n, 'v_n', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path,
                  mesh_function=rmsh.sf)
io.full_print(fsp.sigma_n_12, 'sigma_n_12', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path,
                  mesh_function=rmsh.sf)
    

# print test - end

sys.exit(1)

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


    pr_bc.print_bcs()

    # obtain fsp.sigma_n from fsp.phi by using the definition of fsp.phi
    fsp.sigma_n_12.assign(project(fsp.sigma_n_32 - phi_output, fsp.Q_sigma))

    # Update previous solution
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(fsp.v_n)
    fsp.sigma_n_32.assign(fsp.sigma_n_12)

    pr_sol.print_solution(t, step, vp.dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
