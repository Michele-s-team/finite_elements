"""
This code solves for the dynamics of the Navier Stokes equations with a rigid obstacle which is pinned on one point, and can only rotate about it, on a flat manifold with the monolithic scheme, by defining all fields on discontinuous spaces and using the monolithic approach

NOTE: 
    - in the remeshing block, I do not transfer past fields *_n_1 because they are overwritten right after remeshing
    - if 'remesh_stride' is in 'rpam.parameters', remeshing is done every rpam.parameters['remesh_stride'] steps, while if 'mesh_quality_threshold' is in 'rpam.parameters', remeshing is made when mesh quality drops below 'rpam.parameters['mesh_quality_threshold']'

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:

    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/shape_line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/rigid_obstacle/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_shape_line $MESH_PATH $SOLUTION_PATH
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

import calculus as cal
import continuation as cont
import function as fu
import input_output as io
import mesh.utils as msh
import mesh_quality as msh_qu
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

fi = importlib.import_module(swi.fi)

mesh_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, '../', 'mesh_parameters.csv')) 


print(f'Generating initial mesh ...')

# coordinates of the shape when the shape lies flat (theta_ref = 0)
shape_parametric_form = io.read_function_expresssion(mesh_parameters['shape_parametric_form'])

shape_coordinates_0 = [shape_parametric_form(i/mesh_parameters['N']) for i in range(mesh_parameters['N'])]

# generate the mesh with the shape given by shape_coordinates and write into its mesh_metadata
msh.generate_square_shape_line_mesh(shape_coordinates_0, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory)

print(f'... done.')



# rpam.parameters["theta_0"] is the rotation angle of the shape in the reference configuration 

# trace the coordinates of flat shape vertices by rotating by rpam.parameters["theta_0"] shape_coordinates_0 
shape_coordinates = []
for coordinate in shape_coordinates_0:
    shape_coordinates.append((np.add(mesh_parameters['c'], cal.R(rpam.parameters["theta_0"]).dot(np.subtract(coordinate, mesh_parameters['c'])))).tolist())



dt = rpam.parameters['T'] / rpam.parameters['num_steps']  # time step size

'''
# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                #   'maximum_iterations': 1000000,
                #   'relaxation_parameter': 0.95,
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





print(f'Generating initial mesh ...')

# generate the mesh with the shape and write theta_ref = rpam.parameters["theta_0"] into its mesh_metadata
msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory,
additional_metadata={'phi': rpam.parameters["theta_0"]})


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

class v_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class sigma_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 0

    def value_shape(self):
        return (1,)

# 2.1.1 theta_n, omega_n

fsp.theta_0.assign(Constant(rpam.parameters['theta_0']))
fsp.theta_n_1.assign(Constant(rpam.parameters['theta_0']))

fsp.omega_0.assign(Constant(rpam.parameters['omega_0']))
fsp.omega_n_1.assign(Constant(rpam.parameters['omega_0']))


# 2.1.2 v_n, sigma_n

msh.interpolate_dg(fsp.v_0, v_0_expression())
msh.interpolate_dg(fsp.v_n_1, v_0_expression())

msh.interpolate_dg(fsp.sigma_0, sigma_0_expression())

# 2.1.3 u_n, u_dot_n



# assign _0 fields to `psi` so as to make it start from a good initial guess
fsp.assigner.assign(fsp.psi, [fsp.theta_0, fsp.omega_0, fsp.v_0, fsp.sigma_0, fsp.u_0, fsp.u_dot_0])




#3. Time-stepping

print("Starting time iteration ...", flush=True)

t = 0
step = 0
do_remesh = False

for n in range(rpam.parameters['num_steps']):

    #3.1 Update current time

    t += dt
    step += 1

    #3.2 solve variational problem

    print('Solving monolithic problem ... ')

    if step <= rpam.parameters['n_hold']:
        
        cont.pressure_scale = Constant(0.0)
    
    elif step <= rpam.parameters['n_hold'] + rpam.parameters['n_ramp']:
    
        cont.pressure_scale = Constant((step - rpam.parameters['n_hold']) / rpam.parameters['n_ramp'])
    
    else:
        cont.pressure_scale = Constant(1.0)
        
    vp = importlib.reload(importlib.import_module(swi.vp))  

    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi, parameters=params)

    print('... done.', flush=True)

 
    #3.3 print BCs, ICs, data such as mesh quality, and decompose the deformation field. Note: print_bcs and print_ics must be before the fields update to print the correct residuals of BCs

    #3.3.1 compute mesh quality
    _, _, _, _, u_n_dummy_mesh_quality, _ = fsp.psi.split( deepcopy=True )
    msh_qu.quality = msh.custom_mesh_quality(msh.deform_mesh(rmsh.lmsh.mesh[0], u_n_dummy_mesh_quality))


    #3.3.3 compure BCs, ICs and data
    pr_bc.print_bcs(step)
    pr_ic.print_ics(step)
    pr_da.print_data(step)

    

    # evaluate whether remeshing should be done and write the result in `do_remesh`
    if 'remesh_stride' in rpam.parameters:
        # `rpam.parameters` constains `remesh_stride` -> do remesh every `rpam.parameters['remesh_stride]`

        do_remesh = (step % rpam.parameters['remesh_stride'] == 0)

    elif 'mesh_quality_threshold' in rpam.parameters: 
        # `rpam.parameters` constains `mesh_quality_threshold` -> do remesh when mesh quality drops below `rpam.parameters['mesh_quality_threshold]`

        do_remesh = (msh_qu.quality < rpam.parameters['mesh_quality_threshold'])


    if do_remesh:
        #4. remesh    

        print(f'{col.Fore.CYAN}Remeshing ... {col.Style.RESET_ALL}')

        #4.1 transfer fields

        #4.1.1 Define _old fields that store the last configurations from the last iteration with the previous mesh

        # 4.1.1.1 body
        
        theta_n_old = Function(fsp.Q_theta_n)
        omega_n_old = Function(fsp.Q_omega_n)

        # 4.1.1.2 fluid

        v_n_old = Function(fsp.Q_v_n)
        sigma_n_old = Function(fsp.Q_sigma_n)

        # 4.1.1.3 domain

        u_n_old = Function(fsp.Q_u_n)
        u_dot_n_old = Function(fsp.Q_u_dot_n)


        #4.1.2 Write in the _old fields the configurations form the last iteration with the previous mesh

        #4.1.2.1 unpack the mixed field 
        theta_n_dummy, omega_n_dummy, v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )

        # 4.1.2.2 write

        # 4.1.2.1 body

        theta_n_old.assign(theta_n_dummy)
        omega_n_old.assign(omega_n_dummy)

        # 4.1.2.2 body

        v_n_old.assign(v_n_dummy)
        sigma_n_old.assign(sigma_n_dummy)

        # 4.1.2.3 domain

        u_n_old.assign(u_n_dummy)
        u_dot_n_old.assign(u_dot_n_dummy)


        #4.1.1.2.4 store the measure and marker function of the old mesh (the reference keeps the old MeshFunction and its mesh alive through the module reloads)
        dx_old = rmsh.dx_mesh[0]['dx']
        sf_old = rmsh.sf[0]

         
        #4.2 set the new rotation angle of the polygon for the reference configuration 

        # obtain the angle of the body in the new reference configuration, `theta_ref`, by averaging `theta_n` over the whole mesh (`theta_n` is constant over the mesh)
        theta_ref = msh.average_wrt_measure(fsp.theta_n, rmsh.dx_mesh[0]['dx'])

        # compute the  `shape_coordinates` in the new reference configuration by rotating them with `theta_ref`
        shape_coordinates = []
        for coordinate in shape_coordinates_0:
            shape_coordinates.append(np.add(mesh_parameters['c'], cal.R(theta_ref).dot(np.subtract(coordinate, mesh_parameters['c']))))


        #4.2.1 generate the mesh with the new shape_coordinates

        msh.generate_square_shape_line_mesh(shape_coordinates, os.path.join(rarg.args.input_directory, '../'), rarg.args.input_directory, additional_metadata={'phi': theta_ref})

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


        # 4.4.1 body: there the fields `theta` and `omega` are constant across the mesh, so I simply transfer them with 

        fsp.theta_0.assign(Constant(msh.average_wrt_measure(theta_n_old, dx_old)))
        fsp.omega_0.assign(Constant(msh.average_wrt_measure(omega_n_old, dx_old)))

        # 4.4.2: fluid

        msh.transfer_dg(v_n_old, fsp.v_0, u_n_old, sf_old, rmsh.sf[0])
        msh.transfer_dg(sigma_n_old, fsp.sigma_0, u_n_old, sf_old, rmsh.sf[0])


        # 4.4.3: domain

        fsp.u_0.assign(Constant((0, 0)))
        msh.transfer_dg(u_dot_n_old, fsp.u_dot_0, u_n_old, sf_old, rmsh.sf[0])

        fsp.assigner.assign(fsp.psi, [fsp.theta_0, fsp.omega_0, fsp.v_0, fsp.sigma_0, fsp.u_0, fsp.u_dot_0])

        # sign

        msh.interpolate_dg(fsp.y, fu.identity_expression())

                

        #4.4.3 clean up

        del theta_n_old, omega_n_old, v_n_old, sigma_n_old, u_n_old, u_dot_n_old
        gc.collect()


        print(f'{col.Fore.CYAN}... done.{col.Style.RESET_ALL}')


           

    #5 Update fields

    #5.1 unpack the mixed field 
    theta_n_dummy, omega_n_dummy, v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )

    #5.2 update fields

    fsp.theta_n_1.assign(theta_n_dummy)
    fsp.omega_n_1.assign(omega_n_dummy)

    fsp.v_n_1.assign(v_n_dummy)

    fsp.u_n_1.assign(u_n_dummy)
    fsp.u_dot_n_1.assign(u_dot_n_dummy)

    #5.3 clean up
    del theta_n_dummy, omega_n_dummy, v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy, u_n_dummy_mesh_quality

    #6. print the solution
    if step % rpam.parameters['print_out_stride'] == 0:

        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        
        pr_sol.print_solution(t, step)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)
    

print("... done.", flush=True)

# 7. close files
fi.csvfile_bcs.close()
fi.csvfile_data.close()
fi.csvfile_ics.close()