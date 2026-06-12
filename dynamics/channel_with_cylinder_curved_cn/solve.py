"""
This code solves for the dynamics of the Navier Stokes equations on a fixed, curved manifold with Crank Nicholson discretization scheme

Run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution] T N

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_curved_cn/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $SOLUTION_PATH

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
import switch_problem as swi
import print_out_solution as pr_sol
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10


# set the initial profiles
fsp.v_n_1.interpolate(vp.TangentVelocityExpression(element=fsp.Q_v.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.w.interpolate(vp.NormalVelocityExpression(element=fsp.Q.ufl_element()))
fsp.sigma_n_12.interpolate(vp.SurfaceTensionExpression(element=fsp.Q.ufl_element()))
fsp.sigma_n_32.assign(fsp.sigma_n_12)
fsp.z.interpolate(vp.ManifoldExpression(element=fsp.Q_z.ufl_element()))
fsp.omega.interpolate(vp.OmegaExpression(element=fsp.Q_omega.ufl_element()))

pr_sol.print_z_omega()

params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                'linear_solver': 'default',
                'absolute_tolerance': 1e-10,
                'relative_tolerance': 1e-9,
                'maximum_iterations': 50,
                'relaxation_parameter': None,
                'preconditioner': 'default'
              }
          }

print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['num_steps']):

    # Update current time
    t += vp.dt
    step += 1

    vp = importlib.import_module(swi.vp)

    # step 1: tentative velocity step
    var_pr.solve_vp(vp.F1, fsp.v_, vp.bc_v_, fsp.J_v_, parameters=params)

    # step 2: surface_tension correction step
    var_pr.solve_vp(vp.F2, fsp.phi, vp.bc_phi, fsp.J_phi, parameters=params)

    # step 3: velocity step
    var_pr.solve_vp(vp.F3, fsp.v_n, vp.bc_v_n, fsp.J_v_n, parameters=params)


    pr_bc.print_bcs()

    # obtain fsp.sigma_n from fsp.phi by using the definition of fsp.phi
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - fsp.phi)

    # Update previous solution
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(fsp.v_n)

    fsp.sigma_n_32.assign(fsp.sigma_n_12)

    pr_sol.print_solution(t, step, vp.dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
