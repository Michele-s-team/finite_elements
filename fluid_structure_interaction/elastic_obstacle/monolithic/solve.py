"""
This code solves for the dynamics of the Navier Stokes equations with an elastic obstacle which is pinned on part of its boundary on a flat manifold Crank Nicholson discretization scheme, by defining all fields on discontinuous spaces and using the monolithic approach

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse_circle/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_ellipse_circle $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line $MESH_PATH $SOLUTION_PATH
"""

import colorama as col
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

import continuation as cont
import input_output as io
import mesh.utils as msh
import mesh_quality as msh_qu
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

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

PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')
PETScOptions.set('snes_atol', 1e-12)     
PETScOptions.set('snes_rtol', 1e-12)     
PETScOptions.set('snes_stol', 1e-8)      
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 1000000)         

'''


print(f'Generating initial mesh ...')
# coordinates of the shape when the shape lies flat (theta_ref = 0)
shape_parametric_form = io.read_function_expresssion(mesh_parameters['shape_parametric_form'])

shape_coordinates = [shape_parametric_form(i/mesh_parameters['N']) for i in range(mesh_parameters['N'])]

# generate the mesh with the shape given by shape_coordinates and write into its mesh_metadata
msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)

print(f'... done.')

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


# test patch fields - start
print(f'**** Testing patch ... ')

import solution_paths as solpath


Q_sigma = FunctionSpace(rmsh.lmsh.mesh[0], 'DG', 2)
sigma = Function(Q_sigma)

class sigma_shape_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]+x[1]

    def value_shape(self):
        return (1,)
    
class sigma_square_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]-x[1]**2

    def value_shape(self):
        return (1,)

msh.interpolate_dg(sigma, sigma_shape_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_0_id'])
msh.interpolate_dg(sigma, sigma_square_expression(), rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

io.full_print(sigma, 'sigma_not_patched', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

msh.patch_interface_dofs(sigma, rmsh.sf[0], rmsh.mf_I[0], rmsh.lmsh.parameters['shape_id'], rmsh.lmsh.parameters['sub_mesh_0_0_id'], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

io.full_print(sigma, 'sigma_patched', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])

print(f'**** ... done.')
# test patch fields - end

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

# 0. store metadata

# 0.1 store mesh metadata
mesh_metadata = rmsh.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

# 0.2 store solution metadata
solution_metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'solution_metadata.csv'), solution_metadata)


#1. set the initial profiles

# 1.1 set from expressions
# 

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
        values[0] = rpam.parameters['sigma_r']

    def value_shape(self):
        return (1,)

msh.interpolate_dg(fsp.v_n_1, v_0_expression())
# 

'''
# 1.2 set from files
# 
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'v_n_{rpam.parameters["ic_n"]}.csv'), fsp.v_input)
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'sigma_n_{rpam.parameters["ic_n"]}.csv'), fsp.sigma_input)
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'u_n_{rpam.parameters["ic_n"]}.csv'), fsp.u_input)
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'u_dot_n_{rpam.parameters["ic_n"]}.csv'), fsp.u_dot_input)

# 1.2.1 set v_n_1, u_n_1 and u_dot_n_1 according to the initial condition: in this way, the dynamics will start from where it left off
fsp.v_n_1.assign(fsp.v_input)
fsp.u_n_1.assign(fsp.u_input)
fsp.u_dot_n_1.assign(fsp.u_dot_input)

# 1.2.2 write the read initial condition into psi to let the solver start from a good initial point
fsp.assigner.assign(fsp.psi, [fsp.v_input, fsp.sigma_input, fsp.u_input, fsp.u_dot_input])

#
# '''




#2. Time-stepping

print("Starting time iteration ...", flush=True)

t = 0
step = 0

for n in range(rpam.parameters['num_steps']):

    #2.1 Update current time
    t += dt
    step += 1

    #2.2 solve variational problem


    print('Solving monolithic problem ... ')

    if step <= rpam.parameters['n_hold']:
        cont.pressure_scale = 0.0
    else:
        cont.pressure_scale = 1.0
        
    vp = importlib.reload(importlib.import_module(swi.vp))  # rebuilds F with new pressure_scale

    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi)

    print('... done.', flush=True)

    #2.3 print BCs, ICs and useful data such as mesh quality. Note: print_bcs() and print_ics() must be before the fields update to print the correct residuals of BCs

    _, _, u_n_dummy, _ = fsp.psi.split( deepcopy=True )
    msh_qu.quality = msh.custom_mesh_quality(msh.deform_mesh(rmsh.lmsh.mesh[0], u_n_dummy))

    pr_bc.print_bcs()
    pr_ic.print_ics()
    pr_da.print_data()


    # if msh_qu.quality < rpam.parameters['mesh_quality_threshold']:
    if step > 1:
        # the mesh quality got below the threshold -> remesh 


        print(f'{col.Fore.CYAN}Remeshing ... {col.Style.RESET_ALL}')

        # 1.transfer fields

        # 1.1 Define _old fields that store the last configurations from the last iteration with the previous mesh

        v_n_old = Function(fsp.Q_v_n)
        v_n_1_old = Function(fsp.Q_v_n)
        
        sigma_n_old = Function(fsp.Q_sigma_n)

        u_n_old = Function(fsp.Q_u_n)
        u_dot_n_old = Function(fsp.Q_u_dot_n)
        u_dot_n_1_old = Function(fsp.Q_u_dot_n)

        # 1.2 Write in the _old fields the configurations form the last iteration with the previous mesh

        #1.2.1 unpack the mixed field 
        v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )

        # 1.2.2 write
        v_n_old.assign(v_n_dummy)
        v_n_1_old.assign(fsp.v_n_1)

        sigma_n_old.assign(sigma_n_dummy)

        u_n_old.assign(u_n_dummy)
        u_dot_n_old.assign(u_dot_n_dummy)
        u_dot_n_1_old.assign(fsp.u_dot_n_1)


        #3. trace the coordinates of shape vertices according to the deformation field u_n: these will be the coordinates of the new reference configuration of the shape

        mesh_0_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, f'mesh_{0}', 'mesh_metadata.csv')) 



        shape_coordinates = []
        for i in range(len(mesh_0_parameters["shape_coordinates"])):
            # run through all coordinates of the nodes of mesh[1]

            coordinate = mesh_0_parameters["shape_coordinates"][i]

            # the new reference coordinate is obtained by adding to the previous reference coordinate, the displacement field u_n
            shape_coordinates.append(np.add(
                                        coordinate,
                                        u_n_dummy(coordinate)
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
        pr_ic = importlib.reload(pr_ic)
        pr_da = importlib.reload(pr_da)

        #6. transfer the values stored in the _old fields to the fields defined on the new mesh
        '''
                msh.transfer_dg(v_n_old, fsp.v_n, u_n_old)
                msh.transfer_dg(v_n_1_old, fsp.v_n_1, u_n_old)

                msh.transfer_dg(sigma_n_old, fsp.sigma_n, u_n_old)

                fsp.u_n.assign(Constant((0, 0)))

                msh.transfer_dg(u_dot_n_old, fsp.u_dot_n, u_n_old)
                msh.transfer_dg(u_dot_n_1_old, fsp.u_dot_n_1, u_n_old)
        '''

        #9 clean up

        del v_n_old, v_n_1_old, sigma_n_old, u_n_old, u_dot_n_old, u_dot_n_1_old
        gc.collect()


        print(f'{col.Fore.CYAN}... done.{col.Style.RESET_ALL}')






    #2.4 unpack the mixed field 
    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )

    #2.6 Update fields
    fsp.v_n_1.assign(v_n_dummy)

    fsp.u_n_1.assign(u_n_dummy)
    fsp.u_dot_n_1.assign(u_dot_n_dummy)

    # 2.7 print the solution
    if step % rpam.parameters['print_out_stride'] == 0:

        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        
        pr_sol.print_solution(t, step, dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)
    

print("... done.", flush=True)

