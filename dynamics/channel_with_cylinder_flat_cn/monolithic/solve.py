"""
This code solves for the dynamics of the Navier Stokes equations on a flat manifold with the monolithic scheme, which solves directly for v and p together (no splitting steps)

Run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_flat_cn/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square $MESH_PATH $SOLUTION_PATH
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
import variational_problem.utils as var_pr
import print_out_solution as pr_sol


rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']


# set the initial profiles
# fsp.v_n.interpolate(vp.TangentVelocityExpression(element=fsp.Q_v_n.ufl_element()))
# fsp.v_n_1.assign(fsp.v_n_1)
# fsp.sigma_n.interpolate(vp.SurfaceTensionExpression(element=fsp.Q.ufl_element()))


print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['num_steps']):
    # Update current time
    t += vp.dt
    step += 1

    vp = importlib.import_module(swi.vp)

    # step
    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi)


    pr_bc.print_bcs()


    # print the solution
    v_n_dummy, _ = fsp.psi.split( deepcopy=True )

    # update the solution
    fsp.v_n_1.assign(v_n_dummy)

    if (step % rpam.parameters['print_out_stride']) == 0:

        pr_sol.print_solution(t, step)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
