"""
This code solves for the dynamics of the Navier Stokes equations with a fluid obstacle on a flat manifold, by defining all fields on discontinuous spaces and using the monolithic approach

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:

    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/fluid_obstacle/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line_a $MESH_PATH $SOLUTION_PATH

"""

import colorama as col
import dolfin
from fenics import *
import gc
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.utils as msh
import mesh_quality as msh_qu
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

fi = importlib.import_module(swi.fi)


mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 


dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

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

'''
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'superlu',
        'line_search': 'bt', 
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 1000000,
        'report': True,
    }
}
'''
PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')
PETScOptions.set('snes_atol', 1e-12)     
PETScOptions.set('snes_rtol', 1e-12)     
PETScOptions.set('snes_stol', 1e-8)      
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 1000000)         



print(f'Generating initial mesh ...')
# coordinates of the shape when the shape lies flat (theta_ref = 0)
shape_parametric_form = io.read_function_expresssion(mesh_parameters['shape_parametric_form'])

shape_coordinates = [shape_parametric_form(i/mesh_parameters['N']) for i in range(mesh_parameters['N'])]

# generate the mesh with the shape given by shape_coordinates and write into its mesh_metadata
msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)

print(f'... done.')

'''
# test n_cur - start
import calculus as cal 
import constants.utils as const
import differential_geometry.boundary.geometry as bgeo
import numpy as np
from scipy.optimize import brentq

rmsh = importlib.import_module(swi.rmsh)
sh = importlib.import_module(swi.sh)
import solution_paths as solpath

sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])

 
def theta(s):
    return cal.atan_quad([ sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ])




# s_0 is the value of the curvilinear coordinate at which the polar angle of y(s) with respect to c is 0
theta_0 = theta(0)
# print(f'*** theta(0) = {theta_0 * const.rad_to_deg}')

# 1. Determine  s_0
# 1.1

if (abs(theta_0) < const.epsilon) or (abs(theta_0 - 2*np.pi) < const.epsilon):
    # here s_0 = 1

    theta_0 = 0

    # print(f'** Setting s_0 = 1')

    s_0 = 1

else:
    # theta_0 != 0 -> need to solve for s_0

    # print(f'** Solving for s_0 ... ')

    # 1.1 find two values of s, s_0_L and s_0_R, that bracket s_0, by looking for a pair of values of s, s_i and s_i_plus_1 such that theta(s_i_plus_1) < theta(s_i). This is done by dividing the interval 0 < s < 1 into ~ N_scan interals. Note that N_scan needs to be large enough to resolve the root. 
    for i_s in range(rpam.parameters['N_scan']+1):
        
        if theta((i_s+1)/(rpam.parameters['N_scan']-1)) < theta(i_s/(rpam.parameters['N_scan']-1)): 
            break

    s_0_L = i_s/(rpam.parameters["N_scan"]-1)
    s_0_R = (i_s+1)/(rpam.parameters["N_scan"]-1)

    # print(f's_0 lies betwee {s_0_L} and {s_0_R}')

    # 1.2 solve for s_0 by using s_0_L and s_0_R
    s_0 = brentq(lambda s:  np.arctan((sh.y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (sh.y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0])), a=s_0_L, b=s_0_R)

    # print(f's_0 = {s_0}\t theta(s_0) = {theta(s_0)}')

    # print(f'... done. ')
# 



# return the parameter `s` of the curve `y(s)` corresponding to a point `y` on the plane
# Input values: 
#     - `y`: the point in the reference configuration
# Return values: 
#     - `s`: the value of the parametric coordinate of the curve `y(s)` such that the point y(s) forms the same polar angle as `y` with respect to `c`

def s_y(y):

    # the polar angle that `y` forms with respect to `c``, theta_y is in [0, 2 pi]
    theta_y = cal.atan_quad([ y[0] - rmsh.parameters['c'][0], y[1] - rmsh.parameters['c'][1] ])

    if (abs(theta_y) < const.epsilon) or (abs(theta_y - 2.0 * np.pi) < const.epsilon): 
        # theta_y takes the special values 0 or 2 pi -> set directly s = s_0

        s = s_0

    else:
        # theta_y is not equal to 0 nor to 2 pi -> solve for s

        if 0 < theta_y <= theta_0:
            
            # 0 < theta_y <= theta_0: the solution s must lie in the interval s_0 < s <= 1.0. 
            

            s_L = s_0 + const.epsilon
            s_R = 1 + const.epsilon
            
            s = brentq(lambda s: theta(s) - theta_y, a=s_L, b=s_R)
                            
            # print(f'Case I\n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(s_0+const.epsilon)}\n\ttheta_R = {theta(1 + const.epsilon)}', flush=True)

        else:
            # theta_0 <= theta_y < 2 pi,  the solution s must lie in the interval 0 <= s < s_0

            if theta(0 + const.epsilon) < theta_y: 

                s_L = 0 + const.epsilon

            else:

                s_L = 0

            s_R = s_0 - const.epsilon

            s = brentq(lambda s: theta(s) - theta_y, a=s_L, b=s_R)

    # print(f'\t s = {s}')

    return s


class dyds_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of the parametric coordinate `s` corresponding to `x`
        s = s_y(x)

        values[0] = sh.y_s_dy_ds(s)[1][0]
        values[1] = sh.y_s_dy_ds(s)[1][1]

    def value_shape(self):
        return (2,)
    
class u_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]
        values[1] = x[1]

    def value_shape(self):
        return (2,)


Q_u = VectorFunctionSpace(rmsh.lmsh.mesh[0], 'DG', 2)
Q_dyds = VectorFunctionSpace(rmsh.lmsh.mesh[0], 'DG', 2)

u = Function(Q_u)
dyds = Function(Q_dyds)

msh.interpolate_dg(dyds, dyds_expression())
msh.interpolate_dg(u, u_expression())



n_ref = bgeo.field_facet_normal_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label), rmsh.ds_mesh[0]['dS_shape'], interior=True)

io.full_print(dyds, 'dyds', \
    solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

io.full_print(n_ref, 'n_ref', \
    solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

io.full_print(project(bgeo.n_cur(n_ref, u, dyds), Q_u), 'n_cur', \
    solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

# test n_cur - end
'''



# first load of modules
import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo

fsp = importlib.import_module(swi.fsp)
pr_bc = importlib.import_module(swi.prout_bc)
pr_ic = importlib.import_module(swi.prout_ic)
pr_da = importlib.import_module(swi.prout_da)
pr_sol = importlib.import_module(swi.prout_sol)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

# 1. store metadata

# 1.1 store mesh metadata
mesh_metadata = rmsh.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

# 1.2 store solution metadata
solution_metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'solution_metadata.csv'), solution_metadata)


#2. set the initial profiles

#2.1 set from expressions

# trial analytical expression for a vector
class v_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)

msh.interpolate_dg(fsp.v_n_1, v_0_expression())
# 

'''
# 2.2 set from files
# 
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'v_n_{rpam.parameters["ic_n"]}.csv'), fsp.v_input)
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'sigma_n_{rpam.parameters["ic_n"]}.csv'), fsp.sigma_input)
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'u_n_{rpam.parameters["ic_n"]}.csv'), fsp.u_input)
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'u_dot_n_{rpam.parameters["ic_n"]}.csv'), fsp.u_dot_input)

# 2.2.1 set v_n_1, u_n_1 and u_dot_n_1 according to the initial condition: in this way, the dynamics will start from where it left off
fsp.v_n_1.assign(fsp.v_input)
fsp.u_n_1.assign(fsp.u_input)
fsp.u_dot_n_1.assign(fsp.u_dot_input)

# 2.2.2 write the read initial condition into psi to let the solver start from a good initial point
fsp.assigner.assign(fsp.psi, [fsp.v_input, fsp.sigma_input, fsp.u_input, fsp.u_dot_input])

#
'''



#3. Time-stepping

print("Starting time iteration ...", flush=True)

t = 0
step = 0

for n in range(rpam.parameters['num_steps']):

    #3.1 Update current time

    t += dt
    step += 1

    #3.2 solve variational problem

    print('Solving monolithic problem ... ')

    '''
    if step <= rpam.parameters['n_hold']:
        
        cont.pressure_scale = Constant(0.0)
    
    elif step <= rpam.parameters['n_hold'] + rpam.parameters['n_ramp']:
    
        cont.pressure_scale = Constant((step - rpam.parameters['n_hold']) / rpam.parameters['n_ramp'])
    
    else:
        cont.pressure_scale = Constant(1.0)
    '''
        
    # rebuild F with new pressure_scale
    vp = importlib.reload(importlib.import_module(swi.vp))  

    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi, parameters=params)

    print('... done.', flush=True)

    #3.3 print BCs, ICs, data such as mesh quality, and decompose the deformation field. Note: print_bcs and print_ics must be before the fields update to print the correct residuals of BCs


    '''
    #3.3.1 compute mesh quality
    _, _, u_n_dummy_mesh_quality, _ = fsp.psi.split( deepcopy=True )
    msh_qu.quality = msh.custom_mesh_quality(msh.deform_mesh(rmsh.lmsh.mesh[0], u_n_dummy_mesh_quality))

    #3.3.2 decompose the deformation field

    dec_u = importlib.reload(dec_u) 
    vp_u_0 = importlib.reload(vp_u_0) 

    print('Solving for u_0 ... ')

    var_pr.solve_vp(vp_u_0.F, fsp.u_0, vp_u_0.bcs, fsp.J_u_0)

    print('... done.', flush=True)

    # now that u_0 is known, I set phi_0(y) = y + u_0(y) also in \partial \Omega^y_square
    fsp.phi_0.assign(fsp.y + fsp.u_0)
    '''

    #3.3.3 compure BCs, ICs and data
    pr_bc.print_bcs(step)
    pr_ic.print_ics(step)
    pr_da.print_data(step)

    '''
    if msh_qu.quality < rpam.parameters['mesh_quality_threshold']:
    # if step > 1:

        #4. remesh (the mesh quality got below mesh_quality_threshold ->)


        print(f'{col.Fore.CYAN}Remeshing ... {col.Style.RESET_ALL}')

        #4.1 transfer fields

        #4.1.1 Define _old fields that store the last configurations from the last iteration with the previous mesh

        v_n_old = Function(fsp.Q_v_n)
        v_n_1_old = Function(fsp.Q_v_n)
        
        sigma_n_old = Function(fsp.Q_sigma_n)

        u_n_old = Function(fsp.Q_u_n)
        u_dot_n_old = Function(fsp.Q_u_dot_n)
        u_dot_n_1_old = Function(fsp.Q_u_dot_n)

        phi_n_old = Function(fsp.Q_u_n)
        phi_0_old = Function(fsp.Q_u_n)
        u_0_old = Function(fsp.Q_u_n)

        #4.1.2 Write in the _old fields the configurations form the last iteration with the previous mesh

        #4.1.2.1 unpack the mixed field 
        v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )

        # 4.1.2.2 write
        v_n_old.assign(v_n_dummy)
        v_n_1_old.assign(fsp.v_n_1)

        sigma_n_old.assign(sigma_n_dummy)

        u_n_old.assign(u_n_dummy)
        u_dot_n_old.assign(u_dot_n_dummy)
        u_dot_n_1_old.assign(fsp.u_dot_n_1)

        phi_n_old.assign(project(fsp.y + u_n_dummy, fsp.Q_u_n))
        phi_0_old.assign(fsp.phi_0)
        u_0_old.assign(fsp.u_0)


                
        # 4.1.3 Fields v_n_old, v_n_1_old and sigma_n_old are discontinuous across the shape -> in order to use `transfer` on them, I overwrite their DOFs at the interface belonging to sub_mesh_0_0_id with the respective DOFs at the interface belonging to sub_mesh_0_0_id. In this way, when `transfer` will evaluate v_n_old, v_n_1_old, sigma_n_old ... at a point `x` lying on the interface, it will always use the correct value (the one belonging to sub_mesh_0_1)
        

        msh.overwrite_interface_dofs(v_n_old, rmsh.sf[0], rmsh.mf_I[0], rmsh.lmsh.parameters['shape_id'], rmsh.lmsh.parameters['sub_mesh_0_0_id'], rmsh.lmsh.parameters['sub_mesh_0_1_id'])
        msh.overwrite_interface_dofs(v_n_1_old, rmsh.sf[0], rmsh.mf_I[0], rmsh.lmsh.parameters['shape_id'], rmsh.lmsh.parameters['sub_mesh_0_0_id'], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

        msh.overwrite_interface_dofs(sigma_n_old, rmsh.sf[0], rmsh.mf_I[0], rmsh.lmsh.parameters['shape_id'], rmsh.lmsh.parameters['sub_mesh_0_0_id'], rmsh.lmsh.parameters['sub_mesh_0_1_id'])


        
        # 4.2

        # Right before remesh, the deformation field is u_n, which corresponds to phi_n. 
        # We decompose phi_n into 
        #     - a part `phi_0(y)` that preserves the elastic energy, which is a comination of a rotation and a rigid translation, 
        #     - a part `u'(y)` that cannot be written as a combination of a rotation and a rigid translation. 

        # We trace the coordinates of shape vertices right after remeshing according to phi_0: these will be the coordinates of the new reference configuration of the shape. 

        # Right after remeshing, the iteration starts with reference coordinates y' = phi_0(y), and with a nonzero deformation u'(phi_0^{-1}(y')) with respect to this reference configuration phi_0. 
        

        mesh_0_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, f'mesh_{0}', 'mesh_metadata.csv')) 

        shape_coordinates = []
        for i in range(len(mesh_0_parameters["shape_coordinates"])):
            # run through all coordinates of the nodes of the boundary

            coordinate = mesh_0_parameters["shape_coordinates"][i]

            # the new reference coordinate is obtained by adding to the previous reference coordinate, the displacement field u_0
            # shape_coordinates.append(np.add(
            #                             coordinate,
            #                             fsp.u_0(coordinate)
            #                             ).tolist()
            #                     )
              
            shape_coordinates.append((dec_u.phi_0_expression()(coordinate)).tolist())

        #4.2.1 generate the mesh with the new shape_coordinates

        msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)

        #4.3 reload modules so everything is updated according to the mesh change

        # ----- WARNING : FROM THIS LINE ON, FIELDS RELATIVE TO THE OLD MESH SET UP WILL BE OVERWRITTEN -----
        importlib.reload(geo)
        importlib.reload(rmsh.lmsh)
        importlib.reload(bgeo)
        fsp = importlib.reload(fsp)
        rmsh = importlib.reload(rmsh)
        pr_bc = importlib.reload(pr_bc)
        pr_ic = importlib.reload(pr_ic)
        pr_da = importlib.reload(pr_da)
        pr_sol = importlib.reload(pr_sol)

        #4.4 transfer the values stored in the _old fields to the fields defined on the new mesh

        
        # 4.4.1 Transfer the fields

        # Given that the transformation reference -> current before right before remeshing, phi_n, is decomposed into phi_0 (A) + u' (B), the field are transferred in two steps
        #     A) trasnfer the field with phi_0_old (u_0_0ld)
        #     B) set a nonzero deformation u' with respect to the reference coordinates y'
        

        # 4.4.1.1 Step A): transfer fields with phi_0_old (u_0_0ld)

        msh.transfer(v_n_old, fsp.v_input, u_0_old)
        msh.transfer(v_n_1_old, fsp.v_n_1, u_0_old)

        msh.transfer(sigma_n_old, fsp.sigma_input, u_0_old)


        #4.4.1.2 Step B): set the initial u'

        #4.4.1.2.1 set u_input

        msh.interpolate_dg(fsp.y, fu.identity_expression())

                
        # phi_0_old(y) = y + u_0_old(y)
        # y' = phi_0_old(y)

        # the function phi_n_old_def that satisfies

        # phi_n_old_def(phi_0_old(y)) = phi_n_old(y)
        # phi_n_old_def(y') = phi_n_old(phi_0_old^{-1}(y'))

        # is constructed as

        # phi_n_old_def = fu.deform_function(phi_n_old, u_0_old)
        

        phi_n_old_def = fu.deform_function(phi_n_old, u_0_old)
        phi_n_old_def.set_allow_extrapolation(True)

        #  This implements Eq. (15) in 'Decomposition of deformation field' 
        fsp.u_input.assign(project(phi_n_old_def - fsp.y, fsp.Q_u_n))

        #4.4.1.2.2 set u_dot_input


        # phi_0_old(y) = y + u_0_old(y)
        # y' = phi_0_old(y)

        # the function u_dot_n_old_def that satisfies

        # u_dot_n_old_def(phi_0_old(y)) = u_dot_n_old(y)
        # u_dot_n_old_def(y') = u_dot_n_old(phi_0_old^{-1}(y'))

        # is constructed as

        # u_dot_n_old_def = fu.deform_function(u_dot_n_old, u_0_old)
        

        u_dot_n_old_def = fu.deform_function(u_dot_n_old, u_0_old)
        u_dot_n_old_def.set_allow_extrapolation(True)

        #  This implements Eq. (16) in 'Decomposition of deformation field' 
        fsp.u_dot_input.assign(project(u_dot_n_old_def, fsp.Q_u_dot_n))

        # 4.4.2 write the profiles of fields right after remeshing into the mixed field psi
        
        fsp.assigner.assign(fsp.psi, [fsp.v_input, fsp.sigma_input, fsp.u_input, fsp.u_dot_input])

        #4.4.3 clean up

        del v_n_old, v_n_1_old, sigma_n_old, u_n_old, u_dot_n_old, u_dot_n_1_old, phi_n_old, phi_0_old, u_0_old
        gc.collect()


        print(f'{col.Fore.CYAN}... done.{col.Style.RESET_ALL}')
    '''

    #5 Update fields

    #5.1 unpack the mixed field 
    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy, c_n_dummy, mu_n_dummy, grad_u_n_dummy = fsp.psi.split( deepcopy=True )

    #5.2 update fields
    fsp.v_n_1.assign(v_n_dummy)
    fsp.u_n_1.assign(u_n_dummy)
    fsp.c_n_1.assign(c_n_dummy)

    #5.3 clean up
    del v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy, c_n_dummy, mu_n_dummy, grad_u_n_dummy

    #6. print the solution
    if step % rpam.parameters['print_out_stride'] == 0:

        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        
        pr_sol.print_solution(t, step, dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)
    

print("... done.", flush=True)

# 7. close files
fi.csvfile_bcs.close()
fi.csvfile_data.close()
fi.csvfile_ics.close()