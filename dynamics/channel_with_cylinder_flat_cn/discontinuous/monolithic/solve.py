"""
This code solves for the dynamics of the Navier Stokes equations on a flat  manifold by using discontinuous function spaces and using a monolithic approach (no splitting steps)

Run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/channel_with_cylinder_flat_cn/discontinuous/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH
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
import variational_problem.utils as var_pr

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']

params = {
    'nonlinear_solver': 'newton',
    'newton_solver': {
        'linear_solver': 'superlu',
        'absolute_tolerance': 1e-9,
        'relative_tolerance': 1e-9,
        'maximum_iterations': 50,
    }
}

# set the initial profiles
class v_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    
    

msh.interpolate_dg(fsp.v_n_1, v_0_expression())

print("Starting time iteration ...", flush=True)

# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['num_steps']):
    
    # Update current time
    t += vp.dt
    step += 1

    vp = importlib.import_module(swi.vp)

    # step 1, 2, 3 together
    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi, parameters=params)

    pr_bc.print_bcs()

    v_n_dummy, _ = fsp.psi.split( deepcopy=True )


    # Update previous solution
    fsp.v_n_1.assign(v_n_dummy)

    if (step % rpam.parameters['print_out_stride']) == 0:
    
        pr_sol.print_solution(t, step, vp.dt)


    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)
