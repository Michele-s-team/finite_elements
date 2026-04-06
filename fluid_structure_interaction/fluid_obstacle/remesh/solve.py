'''
This code solves the dynamics of a fluid in a box (A) with a fluid obstacle (B) in the box. 

The problem has three meshes:
- mesh[0]: a 2d mesh given by the box, including the shape in it. This is divided into 
    * sub_mesh[0]: the shape
    * sub_mesh[1]: the surface between the shape boundary and the box. 
- mesh[1]: a 1d mesh given by a line (the boundary of the shape obstacle laid flat on a line)

Run with
    clear; clear; python3 solve.py [name of the variational problem to solve] [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
    
Examples:
     clear; clear; MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/fluid_obstacle/remesh/solution"; rm -rf $MESH_PATH; mkdir $MESH_PATH; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_a $MESH_PATH $SOLUTION_PATH
 '''

import dolfin
from fenics import *
import gc
import importlib
import numpy as np
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi
import variational_problem.utils as var_pr


dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 


dt = rpam.parameters['T'] / rpam.parameters['N']

# create the solution metadata and write it into the output directory 
metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, "solution_metadata.csv"), metadata)


# Use a minimal FEniCS params dict — let PETSc options take over
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'superlu',
        'method': 'newtonls',
        'line_search': 'bt',
        'absolute_tolerance': 1e-10,
        'relative_tolerance': 1e-10,
        'solution_tolerance': 0.0,
        'maximum_iterations': 10000,
        'report': True,
        'error_on_nonconvergence': False,
    }
}


print(f'Generating initial mesh ...')
# coordinates of the shape when the shape lies flat (theta_ref = 0)
shape_parametric_form = io.read_function_expresssion(mesh_parameters['shape_parametric_form'])
shape_coordinates = [shape_parametric_form(i/mesh_parameters['N']) for i in range(mesh_parameters['N'])]

# generate the mesh with the shape given by shape_coordinates and write into its mesh_metadata
msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)
print(f'... done.')



# fist load of modules
import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import print_out_solution as pr_sol
rmsh = importlib.import_module(swi.rmsh)

vp_I = importlib.import_module(swi.vp_I)
vp_D = importlib.import_module(swi.vp_D)
vp_fl_di = importlib.import_module(swi.vp_fluid_di)
vp_fl_sq = importlib.import_module(swi.vp_fluid_sq)
vp_M = importlib.import_module(swi.vp_M)
pr_bc = importlib.import_module(swi.prout_bc)


#0 define classes for initial profiles
# 0.1 I
class U_expression(UserExpression):
    def eval(self, values, x):
        
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    
class ys_expression(UserExpression):
    def eval(self, values, x):

        values[0] = rmsh.lmsh.parameters['c'][0] + rmsh.lmsh.parameters['r'] * np.cos(2 * np.pi * x[0] / rmsh.lmsh.mesh_parameters[1]['L'])
        values[1] = rmsh.lmsh.parameters['c'][1] + rmsh.lmsh.parameters['r'] * np.sin(2 * np.pi * x[0] / rmsh.lmsh.mesh_parameters[1]['L'])
    
    def value_shape(self):
        return (2,)


class psi_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = -2*np.pi*x[0]/rmsh.lmsh.mesh_parameters[1]['L']
    
    def value_shape(self):
        return (1,)
    
class nu_dpsi_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1.0                       
        values[1] = -np.pi/2   

    def value_shape(self):
        return (2,)

# 0.2 fluid square
class v_sq_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)
    

class v_di_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

# 0.3 fluid disk
class sigma_di_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['sigma_di_0']

    def value_shape(self):
        return (1,)
    
class sigma_sq_0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = rpam.parameters['sigma_sq_t'] + rpam.parameters['rho_sq'] * rpam.parameters['g'] * (x[1] - rmsh.lmsh.mesh_parameters[0]['h'])

    def value_shape(self):
        return (1,)


# 1 set initial profiles

# 1.1 I
fsp.U_n_12.interpolate(U_expression(element=fsp.Q_U.ufl_element()))
fsp.ys.interpolate(ys_expression(element=fsp.Q_U.ufl_element()))

fsp.psi_0.interpolate(psi_0_expression(element=fsp.Q_psi_0.ufl_element()))
fsp.nu_and_dpsi_n_12.interpolate(nu_dpsi_expression(element=fsp.Q_nu_and_dpsi.ufl_element()))

# 1.2 fluid square
fsp.v_square_n_1.interpolate(v_sq_0_expression(element=fsp.Q_v_square.ufl_element()))
fsp.v_square_n_2.assign(fsp.v_square_n_1)

fsp.sigma_square_n_12.interpolate(sigma_sq_0_expression(element=fsp.Q_sigma_square.ufl_element()))
fsp.sigma_square_n_32.assign(fsp.sigma_square_n_12)

# 1.3 fluid disk
fsp.v_disk_n_1.interpolate(v_di_0_expression(element=fsp.Q_v_disk.ufl_element()))
fsp.v_disk_n_2.assign(fsp.v_disk_n_1)

fsp.sigma_disk_n_12.interpolate(sigma_di_0_expression(element=fsp.Q_sigma_disk.ufl_element()))
fsp.sigma_disk_n_32.assign(fsp.sigma_disk_n_12)



print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0

for n in range(rpam.parameters['N']):
    # Update current time
    t += dt
    step += 1


    # step 1): solve I problem
    print('Solving I problem ...', flush=True)

    # project v_square_n_1 of the fluid in the square onto (mesh[1]): this velocity will be used in vp_I to make I move
    msh.transfer_2d_to_1d(fsp.v_disk_n_1, fsp.v_disk_n_1_0_0_on_1, rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'], rmsh.lmsh.parameters['shape_id'])

    vp_I = importlib.reload(vp_I)

    # 1.1 solve for U_n_12
    var_pr.solve_vp(vp_I.F_U, fsp.U_n_12, vp_I.bcs_U, fsp.J_U, parameters=params)

    
    #  build a smooth U_n_12 - start
    vp_I.smooth_field_fourier(
        fsp.U_n_12,          
        vp_I.dof_coords, vp_I.dofmap_x, vp_I.dofmap_y,
        rmsh.lmsh.mesh_parameters[1]['L'], n_harmonics=2, target_field=fsp.U_n_12_smooth
    )
    #  build a smooth U_n_12 - end
    

    # 1.2 solve for nu_n_12 and dpsi_n_12
    var_pr.solve_vp(vp_I.F_nu_psi, fsp.nu_and_dpsi_n_12, vp_I.bcs_nu_and_dpsi, fsp.J_nu_and_dpsi, parameters=params)

    # 1.3 solve for mu_n_12
    var_pr.solve_vp(vp_I.F_mu, fsp.mu_n_12, vp_I.bcs_mu, fsp.J_mu, parameters=params)


    print('... done.', flush=True)


    # step 2): solve D problem
    print('Solving D problem ...', flush=True)

    # now that U_n_12 has been computed, compute the new normal
    # POTENTIAL PROBLEM HERE: YOU MAY NEED TO USE A DISCRETE VERSION OF n_ale, using the relation between n and nu
    fsp.n_n_12.assign(project(bgeo.n_ale(fsp.ys, fsp.U_n_12), fsp.Q_U))

    #transfer the new normal it from mesh[1] to sub_mesh[0][0] and write it into n_n_12_1_on_0_0
    # POTENTIAL PROBLEM HERE: YOU MAY NEED TO USE A DISCRETE VERSION OF n_ale, using the relation between n and nu
    msh.transfer_1d_to_2d(fsp.n_n_12, fsp.n_n_12_1_on_0_0, rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'],  rmsh.lmsh.parameters['shape_id'])

    # specify the BC for u_n_di_dot
    fsp.u_n_di_dot_bc_di.assign(project(geo.euclidean_projection(fsp.v_disk_n_1, fsp.n_n_12_1_on_0_0), fsp.Q_u_di_dot))

    #transfer the new normal from mesh[1] to sub_mesh[0][1] and write it into n_n_12_1_on_0_1
    msh.transfer_1d_to_2d(fsp.n_n_12, fsp.n_n_12_1_on_0_1, rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'],  rmsh.lmsh.parameters['shape_id'])
    #transfer v_disk_n_1 form mesh 0 0 to mesh 0 1
    fsp.v_disk_n_1_0_0_on_0_1.assign(project(fsp.v_disk_n_1, fsp.Q_v__square))

    # specify the BC for u_n_sq_dot
    fsp.u_n_sq_dot_bc_di.assign(project(geo.euclidean_projection(fsp.v_disk_n_1_0_0_on_0_1, fsp.n_n_12_1_on_0_1), fsp.Q_u_sq_dot))

    # transfer U_n_12 from mesh[1] to sub_mesh[0][0] and from mesh[1] to sub_mesh[0][1] and write the result in U_n_12_1_on_0_0 and U_n_12_1_on_0_1, respectively: these will be used to set the BCs for u_n_di and u_n_sq in vp_D, respectively
    msh.transfer_1d_to_2d(fsp.U_n_12, fsp.U_n_12_1_on_0_0,  rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'],  rmsh.lmsh.parameters['shape_id'])
    msh.transfer_1d_to_2d(fsp.U_n_12, fsp.U_n_12_1_on_0_1,  rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'],  rmsh.lmsh.parameters['shape_id'])

    vp_D = importlib.reload(vp_D)

    # 2.1) solve for D in square
    var_pr.solve_vp(vp_D.F_u_sq, fsp.u_n_sq, vp_D.bcs_u_sq, fsp.J_u_sq, parameters=params)
    var_pr.solve_vp(vp_D.F_u_sq_dot, fsp.u_n_sq_dot, vp_D.bcs_u_sq_dot, fsp.J_u_dot_sq, parameters=params)

    # 2.2) solve for D in disk
    var_pr.solve_vp(vp_D.F_u_di, fsp.u_n_di, vp_D.bcs_u_di, fsp.J_u_di, parameters=params)
    var_pr.solve_vp(vp_D.F_u_di_dot, fsp.u_n_di_dot, vp_D.bcs_u_di_dot, fsp.J_u_dot_di, parameters=params)

    print('... done.', flush=True)


    # 3) solve for disk fluid 

    print('Solving disk fluid problem ...', flush=True)

    # transfer v_square_n_1 and sigma_square_n_32 (defined on sub_mesh[0][1]) on sub_mesh[0][0], and write the result in v_square_n_1_0_1_on_0_0 and sigma_square_n_32_0_1_on_0_0, respectively
    fsp.v_square_n_1_0_1_on_0_0.assign(project(fsp.v_square_n_1, fsp.Q_u_di_dot))
    fsp.sigma_square_n_32_0_1_on_0_0.assign(project(fsp.sigma_square_n_32, fsp.Q_sigma_disk))

    # transfer mu_n_12 (defined on mesh[1]) on sub_mesh[0][0], in order to compute the Laplace force
    msh.transfer_1d_to_2d(fsp.mu_n_12, fsp.mu_n_12_1_on_0_0,  rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'],  rmsh.lmsh.parameters['shape_id'])


    vp_fl_di = importlib.reload(vp_fl_di)

    # 3.1 solve for v_disk__
    var_pr.solve_vp(vp_fl_di.F_v_disk__, fsp.v_disk__, vp_fl_di.bc_v_disk__, fsp.J_v__disk, parameters=params)

    # 3.2 solve for phi_disk (and omega_disk)
    var_pr.solve_vp(vp_fl_di.F_phi_omega_disk, fsp.phi_omega_disk, vp_fl_di.bc_phi_omega_disk, fsp.J_phi_omega_disk, parameters=params)

    # 3.3 solve for v_disk_n
    var_pr.solve_vp(vp_fl_di.F_v_disk_n, fsp.v_disk_n, vp_fl_di.bc_v_disk_n, fsp.J_v_disk, parameters=params)

    # write into sigma_disk_n_12
    phi_disk_output, omega_disk_output = fsp.phi_omega_disk.split(deepcopy=True)
    fsp.sigma_disk_n_12.assign(fsp.sigma_disk_n_32 - project(phi_disk_output, fsp.Q_sigma_disk))

    print('... done.', flush=True)


    # 4) solve for square fluid 

    print('Solving square fluid problem ...', flush=True)

    # transfer v_disk_n (defined on sub_mesh[0][0]) on sub_mesh[0][1], and write the result in v_disk_n_0_0_on_0_1: v_disk_n_0_0_on_0_1 will be used as a BC in vp_fl_sq
    fsp.v_disk_n_0_0_on_0_1.assign(project(fsp.v_disk_n, fsp.Q_v__square))

    vp_fl_sq = importlib.reload(vp_fl_sq)

    # 4.1 solve for v_square__
    var_pr.solve_vp(vp_fl_sq.F_v_square__, fsp.v_square__, vp_fl_sq.bc_v_square__, fsp.J_v__square, parameters=params)

    # 4.2 solve for phi_square
    var_pr.solve_vp(vp_fl_sq.F_phi_square, fsp.phi_square, vp_fl_sq.bc_phi_square, fsp.J_phi_square, parameters=params)

    # 4.3 solve for v_square_n
    var_pr.solve_vp(vp_fl_sq.F_v_square_n, fsp.v_square_n, vp_fl_sq.bc_v_square_n, fsp.J_v_square, parameters=params)

    # write into sigma_square_n_12
    fsp.sigma_square_n_12.assign(fsp.sigma_square_n_32 - fsp.phi_square)

    print('... done.', flush=True)


    # 5) solve for M

    print('Solving M problem ...', flush=True)

    vp_M = importlib.reload(vp_M)

    # solve for c_n
    var_pr.solve_vp(vp_M.F_c, fsp.c_n, vp_M.bc_M, fsp.J_c, parameters=params)
    
    print('... done.', flush=True)


    # print out the residuals of BCs
    # note: print_bcs() must be before the fields update to print the correct residuals of BCs
    if step % rpam.parameters['print_out_stride'] == 0:

        pr_bc.print_bcs()


    mesh_0_0_quality = msh.custom_mesh_quality(msh.deform_mesh(rmsh.lmsh.sub_meshes[0][0], fsp.u_n_di))
    mesh_0_1_quality = msh.custom_mesh_quality(msh.deform_mesh(rmsh.lmsh.sub_meshes[0][1], fsp.u_n_sq))
    mesh_quality = min(mesh_0_0_quality, mesh_0_1_quality)


    # if mesh_quality < rpam.parameters['mesh_quality_threshold']:
    if True:
    # if step % 5 == True:

        mesh_1_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, f'mesh_{1}', 'mesh_metadata.csv')) 


        # the mesh quality got below the threshold -> remesh 
        
        # 1.transfer fields

        # 1.1 Define _old fields that store the last configurations from the last iteration with the previous mesh

        # 1.1.1 disk fluid
        v_di_n_old = Function(fsp.Q_v_disk)
        v_di_n_1_old = Function(fsp.Q_v_disk)
        v_di_n_2_old = Function(fsp.Q_v_disk)

        v_di__old = Function(fsp.Q_v__disk)

        sigma_di_n_12_old = Function(fsp.Q_sigma_disk)
        sigma_di_n_32_old = Function(fsp.Q_sigma_disk)

        phi_disk_old = Function(fsp.Q_phi_disk)
        omega_disk_old = Function(fsp.Q_omega_disk)


        # 1.1.2 square fluid
        v_sq_n_old = Function(fsp.Q_v_square)
        v_sq_n_1_old = Function(fsp.Q_v_square)
        v_sq_n_2_old = Function(fsp.Q_v_square)

        v_sq__old = Function(fsp.Q_v__square)

        sigma_sq_n_12_old = Function(fsp.Q_sigma_square)
        sigma_sq_n_32_old = Function(fsp.Q_sigma_square)

        phi_sq_old = Function(fsp.Q_sigma_square)


        # 1.1.3 D

        # 1.1.3.1 disk
        u_n_di_old = Function(fsp.Q_u_di)
        u_n_1_di_old = Function(fsp.Q_u_di)
        u_n_2_di_old = Function(fsp.Q_u_di)

        u_n_di_dot_old = Function(fsp.Q_u_di_dot)
        u_n_1_di_dot_old = Function(fsp.Q_u_di_dot)
        u_n_2_di_dot_old = Function(fsp.Q_u_di_dot)

        # 1.1.3.2 square
        u_n_sq_old = Function(fsp.Q_u_sq)
        u_n_1_sq_old = Function(fsp.Q_u_sq)
        u_n_2_sq_old = Function(fsp.Q_u_sq)

        u_n_sq_dot_old = Function(fsp.Q_u_sq_dot)
        u_n_1_sq_dot_old = Function(fsp.Q_u_sq_dot)
        u_n_2_sq_dot_old = Function(fsp.Q_u_sq_dot)


        # 1.1.4 I

        U_n_12_old = Function(fsp.Q_U)
        U_n_32_old = Function(fsp.Q_U)

        ys_U_n_12_old = Function(fsp.Q_U)

        mu_n_12_old = Function(fsp.Q_mu)



        # 1.1.5 M

        c_n_old = Function(fsp.Q_c)
        c_n_1_old = Function(fsp.Q_c)


        
        # 1.2 Write in the _old fields the configurations form the last iteration with the previous mesh

        # 1.2.1 disk fluid

        v_di_n_old.assign(fsp.v_disk_n)
        v_di_n_1_old.assign(fsp.v_disk_n_1)
        v_di_n_2_old.assign(fsp.v_disk_n_2)

        v_di__old.assign(fsp.v_disk__)

        sigma_di_n_12_old.assign(fsp.sigma_disk_n_12)
        sigma_di_n_32_old.assign(fsp.sigma_disk_n_32)

        phi_disk_output, omega_disk_output = fsp.phi_omega_disk.split(deepcopy=True)
        phi_disk_old.assign(phi_disk_output)
        omega_disk_old.assign(omega_disk_output)

        # 1.2.2 square fluid

        v_sq_n_old.assign(fsp.v_square_n)
        v_sq_n_1_old.assign(fsp.v_square_n_1)
        v_sq_n_2_old.assign(fsp.v_square_n_2)

        v_sq__old.assign(fsp.v_square__)

        sigma_sq_n_12_old.assign(fsp.sigma_square_n_12)
        sigma_sq_n_32_old.assign(fsp.sigma_square_n_32)

        phi_sq_old.assign(fsp.phi_square)
        
        # 1.2.3 D

        # 1.2.3.1 disk

        u_n_di_old.assign(fsp.u_n_di)
        u_n_1_di_old.assign(fsp.u_n_1_di)
        u_n_2_di_old.assign(fsp.u_n_2_di)

        u_n_di_dot_old.assign(fsp.u_n_di_dot)
        u_n_1_di_dot_old.assign(fsp.u_n_1_di_dot)
        u_n_2_di_dot_old.assign(fsp.u_n_2_di_dot)

        # 1.2.3.2 square

        u_n_sq_old.assign(fsp.u_n_sq)
        u_n_1_sq_old.assign(fsp.u_n_1_sq)
        u_n_2_sq_old.assign(fsp.u_n_2_sq)

        u_n_sq_dot_old.assign(fsp.u_n_sq_dot)
        u_n_1_sq_dot_old.assign(fsp.u_n_1_sq_dot)
        u_n_2_sq_dot_old.assign(fsp.u_n_2_sq_dot)

        # 1.2.4 D

        U_n_12_old.assign(fsp.U_n_12)
        U_n_32_old.assign(fsp.U_n_32)

        ys_U_n_12_old.assign(fsp.ys + fsp.U_n_12)

        mu_n_12_old.assign(fsp.mu_n_12)

        # 1.2.5 M

        c_n_old.assign(fsp.c_n)
        c_n_1_old.assign(fsp.c_n_1)


        #3. trace the coordinates of shape vertices according to the deformation field U_n_12: these will be the coordinates of the new reference configuration of the shape
        shape_coordinates = []
        for i in range(len(mesh_1_parameters["coordinates"])-1):
            # run through all coordinates of the nodes of mesh[1]

            coordinate = mesh_1_parameters["coordinates"][i]

            # the new reference coordinate is obtained by adding to the previous reference coordinate, the displacement field
            shape_coordinates.append(np.add(
                                        msh.map_1d_to_2d(coordinate,  rmsh.lmsh.mesh[0], rmsh.mf[0], rmsh.lmsh.mesh_parameters[0]['shape_coordinates'], rmsh.lmsh.mesh_parameters[0]['shape_id']),
                                        fsp.U_n_12(coordinate)
                                        ).tolist()
                                )   

        #4. generate the mesh with the new shape_coordinates
        msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)


        #5. reload modules so everything is updated according to the mesh change
        importlib.reload(geo)
        importlib.reload(rmsh.lmsh)
        importlib.reload(bgeo)
        fsp = importlib.reload(fsp)
        rmsh = importlib.reload(rmsh)
        pr_bc = importlib.reload(pr_bc)

        #6. reset cleanly solver parameters 


        #7. transfer the values stored in the _old fields to the fields defined on the new mesh

        # 7.1 fluid in disk
        msh.transfer(v_di_n_old, fsp.v_disk_n, u_n_di_old)
        msh.transfer(v_di_n_1_old, fsp.v_disk_n_1, u_n_di_old)
        msh.transfer(v_di_n_2_old, fsp.v_disk_n_2, u_n_di_old)

        msh.transfer(v_di__old, fsp.v_disk__, u_n_di_old)

        msh.transfer(sigma_di_n_12_old, fsp.sigma_disk_n_12, u_n_di_old)
        msh.transfer(sigma_di_n_32_old, fsp.sigma_disk_n_32, u_n_di_old)

        msh.transfer(phi_disk_old, fsp.phi_disk_aux, u_n_di_old)
        msh.transfer(omega_disk_old, fsp.omega_disk_aux, u_n_di_old)
        fsp.assigner_phi_omega_disk.assign(fsp.phi_omega_disk, [fsp.phi_disk_aux, fsp.omega_disk_aux])

        # 7.2 fluid in square
        msh.transfer(v_sq_n_old, fsp.v_square_n, u_n_sq_old)
        msh.transfer(v_sq_n_1_old, fsp.v_square_n_1, u_n_sq_old)
        msh.transfer(v_sq_n_2_old, fsp.v_square_n_2, u_n_sq_old)

        msh.transfer(v_sq__old, fsp.v_square__, u_n_sq_old)

        msh.transfer(sigma_sq_n_12_old, fsp.sigma_square_n_12, u_n_sq_old)
        msh.transfer(sigma_sq_n_32_old, fsp.sigma_square_n_32, u_n_sq_old)

        msh.transfer(phi_sq_old, fsp.phi_square, u_n_sq_old)

        # 7.3 D

        # 7.3.1 disk

        # given that I am starting at the (new) reference configuration, I set the displacement fields to zero 
        fsp.u_n_di.assign(Constant((0, 0)))
        fsp.u_n_1_di.assign(Constant((0, 0)))
        fsp.u_n_2_di.assign(Constant((0, 0)))

        msh.transfer(u_n_di_dot_old, fsp.u_n_di_dot, u_n_di_old)
        msh.transfer(u_n_1_di_dot_old, fsp.u_n_1_di_dot, u_n_di_old)
        msh.transfer(u_n_2_di_dot_old, fsp.u_n_2_di_dot, u_n_di_old)   

        # 7.3.2 square

        # given that I am starting at the (new) reference configuration, I set the displacement fields to zero 
        fsp.u_n_sq.assign(Constant((0, 0)))
        fsp.u_n_1_sq.assign(Constant((0, 0)))
        fsp.u_n_2_sq.assign(Constant((0, 0)))

        msh.transfer(u_n_sq_dot_old, fsp.u_n_sq_dot, u_n_sq_old)
        msh.transfer(u_n_1_sq_dot_old, fsp.u_n_1_sq_dot, u_n_sq_old)
        msh.transfer(u_n_2_sq_dot_old, fsp.u_n_2_sq_dot, u_n_sq_old)   

        # 7.4 I

        # 7.4.1 given that I am starting at the (new) reference configuration, I set the displacement fields to zero 
        fsp.U_n_12.assign(Constant((0, 0)))
        fsp.U_n_32.assign(Constant((0, 0)))

        #7.4.2 given that psi_0 has been recreated from scratch, it is set to 0 -> re-set the correct profile in it
        fsp.psi_0.interpolate(psi_0_expression(element=fsp.Q_psi_0.ufl_element()))
     
        # 7.4.3 set the new ys equal to [the old ys] + [the old U_n_12]
        msh.transfer_1d(ys_U_n_12_old, fsp.ys)

        #7.4.4 given that nu_and_psi_n_12 has been recreated from scratch, is it set to 0, 0 ->  set a reasonable initial guess into nu_and_dpsi_n_12
        fsp.nu_and_dpsi_n_12.interpolate(nu_dpsi_expression(element=fsp.Q_nu_and_dpsi.ufl_element()))

        # 7.4.5 write the new mu_n_12 after remeshing: this may provide a good initial guess when solving for mu_n_12 after remeshing
        msh.transfer_1d(mu_n_12_old, fsp.mu_n_12)



        # 7.5 M
        msh.transfer(c_n_old, fsp.c_n, u_n_sq_old)
        msh.transfer(c_n_1_old, fsp.c_n_1, u_n_sq_old)


        #8. call print_remesh to print out the remeshing info

        pr_sol.print_remesh(step, mesh_quality)

    
        #9 clean up

        # 9.1 disk and square fluid
        del v_di_n_old, v_di_n_1_old, v_di_n_2_old, v_sq_n_old, v_sq_n_1_old, v_sq_n_2_old
        del v_di__old, v_sq__old
        del sigma_di_n_12_old, sigma_di_n_32_old, sigma_sq_n_12_old, sigma_sq_n_32_old
        del phi_disk_old, omega_disk_old

        # 9.2 D
        del u_n_di_old, u_n_1_di_old, u_n_2_di_old, u_n_sq_old, u_n_1_sq_old, u_n_2_sq_old
        del u_n_di_dot_old, u_n_1_di_dot_old, u_n_2_di_dot_old, u_n_sq_dot_old, u_n_1_sq_dot_old, u_n_2_sq_dot_old

        # 9.3 I
        del U_n_12_old, U_n_32_old, ys_U_n_12_old, mu_n_12_old

        # 9.4 M
        del c_n_old, c_n_1_old

        gc.collect()
        
        print(f'**** ... done. ')


    
    # update the fields
    # 1) I 

    fsp.U_n_32.assign(fsp.U_n_12)

    # 2) D

    # 2.1) disk
    fsp.u_n_2_di.assign(fsp.u_n_1_di)
    fsp.u_n_1_di.assign(fsp.u_n_di)

    fsp.u_n_2_di_dot.assign(fsp.u_n_1_di_dot)
    fsp.u_n_1_di_dot.assign(fsp.u_n_di_dot)

    # 2.2) square
    fsp.u_n_2_sq.assign(fsp.u_n_1_sq)
    fsp.u_n_1_sq.assign(fsp.u_n_sq)

    fsp.u_n_2_sq_dot.assign(fsp.u_n_1_sq_dot)
    fsp.u_n_1_sq_dot.assign(fsp.u_n_sq_dot)


    # 3) disk fluid 
    fsp.v_disk_n_2.assign(fsp.v_disk_n_1)
    fsp.v_disk_n_1.assign(fsp.v_disk_n)

    fsp.sigma_disk_n_32.assign(fsp.sigma_disk_n_12)


    # 4) square fluid 
    fsp.v_square_n_2.assign(fsp.v_square_n_1)
    fsp.v_square_n_1.assign(fsp.v_square_n)

    fsp.sigma_square_n_32.assign(fsp.sigma_square_n_12)

    # 5) M

    fsp.c_n_1.assign(fsp.c_n)



    # print out the solution
    if step % rpam.parameters['print_out_stride'] == 0:
        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output

        pr_sol.print_solution(t, step)

        # generate the mesh with the current shape_coordinates and store it into rarg.args.input_directory/n_[step]/
        msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), os.path.join(solpath.snapshots_path, 'mesh', f'n_{step}'))

    # print mesh quality at each step
    pr_sol.print_data(step, mesh_quality)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)

print("... done.", flush=True)

pr_sol.remesh_csvfile.close()
pr_sol.data_csvfile.close()
