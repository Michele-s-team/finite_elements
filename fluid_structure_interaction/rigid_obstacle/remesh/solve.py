"""
This code solves for the dynamics of the Navier Stokes equations with a rigid obstacle which can rotate about a fixed point, by allowing for remeshing,
 on a flat manifold Crank Nicholson discretization scheme

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    clear; clear; MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/rigid_obstacle/remesh/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_polygon $MESH_PATH $SOLUTION_PATH

Note that all sections of the code which need to be changed when an external parameter (e.g., the inflow velocity, the length of the rectangle, etc...) is changed are bracketed by
#CHANGE PARAMETERS HERE
"""

import dolfin
from fenics import *
import importlib
import numpy as np
import os
import shutil
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
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


######
mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 

focus = np.subtract(mesh_parameters["c"], [np.sqrt(mesh_parameters["a"] ** 2 - mesh_parameters["b"] ** 2), 0])

# trace the coordinates of flat polygon vertices
polygon_coordinates = []
for i in range(mesh_parameters['N']-1):
    polygon_coordinates.append(
        list(np.add(
            mesh_parameters['c'],
            [mesh_parameters['a'] * np.cos(2.0*np.pi*i/(mesh_parameters['N']-1)),
            mesh_parameters['b'] * np.sin(2.0*np.pi*i/(mesh_parameters['N']-1))] 
            ))
    )
print(f'flat polygon_coordinates = {polygon_coordinates}')

# generate the mesh with the polygon
msh.generate_square_polygon_mesh(polygon_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)

###### 

import function_spaces as fsp


# initialize values
fsp.theta_n = rpam.parameters["theta_0"]
fsp.omega_n = rpam.parameters["omega_0"]
fsp.theta_n_1 = rpam.parameters["theta_0"]
fsp.omega_n_1 = rpam.parameters["omega_0"]



rmsh = importlib.import_module(swi.rmsh)
ap_polygon = importlib.import_module(swi.ap_polygon)
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
    ap_polygon = importlib.reload(ap_polygon)

    fsp.theta_n = fsp.theta_n_1 + dt * fsp.omega_n_1
    fsp.omega_n = fsp.omega_n_1 + dt / rpam.parameters["I_ellipse"] * ap_polygon.M_ellipse
    print('... done.', flush=True)

    # step 2): update u and u_dot (mesh problem)
    print('Solving mesh problem ...', flush=True)

    vp_mesh = importlib.reload(vp_mesh)

    # solve for u and u_dot
    var_pr.solve_vp(vp_mesh.F_u, fsp.u_n, vp_mesh.bcs, fsp.J_u, params)
    var_pr.solve_vp(vp_mesh.F_u_dot, fsp.u_dot_n, vp_mesh.bcs_dot, fsp.J_u_dot, params)

    print('... done.', flush=True)

    # step 3) update v_n and sigma_n_12 (fluid problem)
    print('Solving fluid problem ...', flush=True)

    vp_fluid = importlib.reload(vp_fluid)

    # step 3.1: approximate velocity step
    var_pr.solve_vp(vp_fluid.F_v_, fsp.v_, vp_fluid.bc_v_, fsp.J_v_)

    # step 3.2: surface_tension correction step
    var_pr.solve_vp(vp_fluid.F_phi, fsp.phi, vp_fluid.bc_phi, fsp.J_phi)

    # step 3.3: velocity step
    var_pr.solve_vp(vp_fluid.F_v_n, fsp.v_n, vp_fluid.bc_v_n, fsp.J_v_n)

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
