"""
This code solves for the dynamics of the Navier Stokes equations with an elastic obstacle which is pinned on part of its boundary on a flat manifold Crank Nicholson discretization scheme, by defining all fields on discontinuous spaces and using the monolithic approach

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/ellipse_circle/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/monolithic/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_ellipse_circle $MESH_PATH $SOLUTION_PATH
"""

import dolfin
from fenics import *
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import continuation as cont
import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import print_out_ic as pr_ic
import print_out_data as pr_data
import print_out_solution as pr_sol
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr


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

pr_bc = importlib.import_module(swi.prout_bc)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

'''
# test read iniital profiles - start
import solution_paths as solpath
import numpy as np


class t_expression(UserExpression):
    def eval(self, values, x):

        values[0] = np.cos(x[0]-x[1])
        values[1] = 1
        values[2] = np.cos(x[0]+x[1]**2)
        values[3] = 3

    def value_shape(self):
        return (2, 2)
    
msh.interpolate_dg(fsp.t_output, t_expression())

io.full_print(fsp.t_output, 't_output', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)


io.read_dg_field_from_csv_file(f'/home/fenics/shared/fluid_structure_interaction/elastic_obstacle/monolithic/solution/snapshots/csv/t_output.csv', fsp.t_input)

io.full_print(fsp.t_input, 't_input', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf)

sys.exit(1)
# test read initial profile - end
'''


# 0. store metadata

# 0.1 store mesh metadata
mesh_metadata = rmsh.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'mesh_metadata.csv'), mesh_metadata)

# 0.2 store solution metadata
solution_metadata = rpam.parameters.copy()
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'solution_metadata.csv'), solution_metadata)


#1. set the initial profiles

# 1.1 set from expressions
'''
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
'''

# 1.2 set from files
io.read_dg_field_from_csv_file(os.path.join(rpam.parameters['ic_path'], f'sigma_n_{rpam.parameters["ic_n"]}.csv'), fsp.sigma_input)


sys.exit(1)


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

    # 
    import ufl as ufl
    import physics.fluid_mechanics as flu
    i, j, k, l, m = ufl.indices(5)

    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )


    print("||sigma_n|| at interface:", 
        assemble(fsp.sigma_n(vp.sub_mesh_1_label)**2 * rmsh.dS_ellipse)**0.5)
    print("||viscous traction||:", 
        assemble(flu.sigma_ale_no_pressure(
            v_n_dummy(vp.sub_mesh_1_label), Constant(0), 
            u_n_dummy(vp.sub_mesh_1_label), rpam.parameters['mu_fluid']
        )[i,k] * flu.sigma_ale_no_pressure(
            v_n_dummy(vp.sub_mesh_1_label), Constant(0), 
            u_n_dummy(vp.sub_mesh_1_label), rpam.parameters['mu_fluid']
        )[i,k] * rmsh.dS_ellipse)**0.5)
    print("<u_n^2> at interface:", 
        assemble(msh.average(fsp.u_n[i]*fsp.u_n[i]) * rmsh.dS_ellipse)**0.5)
    # 

    print('... done.', flush=True)

    #2.3 note: print_bcs() and print_ics() must be before the fields update to print the correct residuals of BCs
    pr_bc.print_bcs()
    pr_ic.print_ics()
    pr_data.print_data()

    #2.4 unpack the mixed field 
    v_n_dummy, sigma_n_dummy, u_n_dummy, u_dot_n_dummy = fsp.psi.split( deepcopy=True )


    #2.6 Update fields
    fsp.v_n_1.assign(v_n_dummy)

    fsp.u_n_2.assign(fsp.u_n_1)
    fsp.u_n_1.assign(u_n_dummy)

    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(u_dot_n_dummy)

    # 2.7 print the solution
    if step % rpam.parameters['print_out_stride'] == 0:

        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        
        pr_sol.print_solution(t, step, dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.parameters['T'])), flush=True)
    

print("... done.", flush=True)

