"""
This code solves for the dynamics of the Navier Stokes equations with a rigid obstacle which can rotate about a fixed point, by allowing for remeshing,
 on a flat manifold Crank Nicholson discretization scheme

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    clear; clear; MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/polygon/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/rigid_obstacle/remesh/solution"; rm -rf $MESH_PATH; mkdir $MESH_PATH; rm -rf $SOLUTION_PATH; python3 solve.py square_polygon $MESH_PATH $SOLUTION_PATH

Note that all sections of the code which need to be changed when an external parameter (e.g., the inflow velocity, the length of the rectangle, etc...) is changed are bracketed by
#CHANGE PARAMETERS HERE
"""

import dolfin
from fenics import *
import importlib
import numpy as np
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import calculus as cal
import elasticity as ela
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi
import variational_problem.utils as var_pr

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 

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

# trial analytical expression for a vector
class v_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


# trial analytical expression for the  surface tension sigma(x,y)
class sigma_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)
    

# focal point of the ellipse
f = cal.ellipse_focal_points(mesh_parameters['a'], mesh_parameters['b'], mesh_parameters['c'])[0]
# coordinates of the ellipse when the ellipse lies flat (theta_ref = 0)
polygon_coordinates_flat = cal.points_ellipse(mesh_parameters['a'], mesh_parameters['b'], mesh_parameters['c'], mesh_parameters['N'])




# theta_ref is the rotation angle of the polygon in the reference configuration 
theta_ref = rpam.parameters["theta_0"]

# trace the coordinates of flat polygon vertices by rotating by theta_ref polygon_coordinates_flat 
polygon_coordinates = []
for coordinate in polygon_coordinates_flat:
    polygon_coordinates.append(np.add(f, cal.R(theta_ref).dot(np.subtract(coordinate, f))))

# generate the mesh with the polygon and write theta_ref into its mesh_metadata
msh.generate_square_polygon_mesh(polygon_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory,
additional_metadata={'phi': theta_ref})





import function_spaces as fsp
import print_out_solution as pr_sol

#set initial profiles and values
fsp.theta_n = rpam.parameters["theta_0"]
fsp.omega_n = rpam.parameters["omega_0"]
fsp.theta_n_1 = rpam.parameters["theta_0"]
fsp.omega_n_1 = rpam.parameters["omega_0"]

fsp.v_n_1.interpolate(v_0_expression(element=fsp.Q_v.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.sigma_n_12.interpolate(sigma_0_expression(element=fsp.Q_phi.ufl_element()))
fsp.sigma_n_32.assign(fsp.sigma_n_12)

# fist load of modules
rmsh = importlib.import_module(swi.rmsh)
ap_polygon = importlib.import_module(swi.ap_polygon)
vp_fluid = importlib.import_module(swi.vp_fluid)
vp_mesh = importlib.import_module(swi.vp_mesh)
pr_bc = importlib.import_module(swi.prout_bc)


# block with the same mesh - start

fsp = importlib.reload(fsp)
rmsh = importlib.reload(rmsh)
pr_bc = importlib.reload(pr_bc)


# Time-stepping
print("Starting time iteration ...", flush=True)

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

    if step % rpam.parameters['remesh_stride']:

        # remesh 

        print(f'**** Remeshing ... ')

        # 1. Define fields that store the last configurations from the iteration with the previous mesh, and store in to them these configurations

        # 1.1
        v_n_old = Function(fsp.Q_v)
        v_n_1_old = Function(fsp.Q_v)
        v_n_2_old = Function(fsp.Q_v)

        v__old = Function(fsp.Q_v_)

        sigma_n_12_old = Function(fsp.Q_phi)
        sigma_n_32_old = Function(fsp.Q_phi)

        phi_old = Function(fsp.Q_phi)

        u_n_old = Function(fsp.Q_u)
        u_n_1_old = Function(fsp.Q_u)
        u_n_2_old = Function(fsp.Q_u)

        u_dot_n_old = Function(fsp.Q_u_dot)
        u_dot_n_1_old = Function(fsp.Q_u_dot)
        u_dot_n_2_old = Function(fsp.Q_u_dot)


        # 1.2
        v_n_old.assign(fsp.v_n)
        v_n_1_old.assign(fsp.v_n_1)
        v_n_2_old.assign(fsp.v_n_2)

        v__old.assign(fsp.v_)

        sigma_n_12_old.assign(fsp.sigma_n_12)
        sigma_n_32_old.assign(fsp.sigma_n_32)

        phi_old.assign(fsp.phi)

        u_n_old.assign(fsp.u_n)
        u_n_1_old.assign(fsp.u_n_1)
        u_n_2_old.assign(fsp.u_n_2)

        u_dot_n_old.assign(fsp.u_dot_n)
        u_dot_n_1_old.assign(fsp.u_dot_n_1)
        u_dot_n_2_old.assign(fsp.u_dot_n_2)

        #2. set the new rotation angle of the polygon for the reference configuration 
        theta_ref = fsp.theta_n

        #2. trace the coordinates of polygon vertices with the new theta_ref polygon_coordinates_flat 
        polygon_coordinates = []
        for coordinate in polygon_coordinates_flat:
            polygon_coordinates.append(np.add(f, cal.R(theta_ref).dot(np.subtract(coordinate, f))))



        # 3. generate the mesh with the polygon and write theta_ref into its mesh_metadata
        msh.generate_square_polygon_mesh(polygon_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory,
        additional_metadata={'phi': theta_ref})

        # 4. reload modules so everything is updated to the new mesh
        importlib.reload(rmsh.lmsh)
        rmsh = importlib.reload(rmsh)
        fsp = importlib.reload(fsp)

        # 5. transfer the values stored in the _old fields to the fields defined on the new mesh

        msh.transfer(v_n_old, fsp.v_n, u_n_old)
        msh.transfer(v_n_1_old, fsp.v_n_1, u_n_old)
        msh.transfer(v_n_2_old, fsp.v_n_2, u_n_old)

        msh.transfer(v__old, fsp.v_, u_n_old)

        msh.transfer(sigma_n_12_old, fsp.sigma_n_12, u_n_old)
        msh.transfer(sigma_n_32_old, fsp.sigma_n_32, u_n_old)

        msh.transfer(phi_old, fsp.phi, u_n_old)

        msh.transfer(u_n_old, fsp.u_n, u_n_old)
        msh.transfer(u_n_1_old, fsp.u_n_1, u_n_old)
        msh.transfer(u_n_2_old, fsp.u_n_2, u_n_old)

        msh.transfer(u_dot_n_old, fsp.u_n, u_dot_n_old)
        msh.transfer(u_dot_n_1_old, fsp.u_n_1, u_dot_n_old)
        msh.transfer(u_dot_n_2_old, fsp.u_n_2, u_dot_n_old)


        '''   
        io.full_print(fsp.sigma_n_12, 'sigma_n_12_new', \
                        solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
        io.full_print_deformed(sigma_n_12_old, u_n_old, 'sigma_n_12_old', \
                        solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

        io.full_print(fsp.v_n, 'v_n_new', \
                        solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
        io.full_print_deformed(v_n_old, u_n_old, 'v_n_old', \
                        solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)

        io.full_print(fsp.sigma_stress_n, 'sigma_stress_n_new', \
                        solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
        io.full_print_deformed(sigma_stress_n_old, u_n_old, 'sigma_stress_n_old', \
                        solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path)
        '''        

        print(f'**** ... done. ')
    

    
    #update the fields
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


# block with the same mesh - end
