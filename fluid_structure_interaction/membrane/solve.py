"""
This code solves for the dynamics of the Navier Stokes equations for a fluid in a square whose top edge is a membrane. The coupled dynamics of  membrane, fluid and of the fictitious elastic body (which defines the region where the fluid moves) are solved. 

run with:
    rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

Examples:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/membrane/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_line_a $MESH_PATH $SOLUTION_PATH
"""
import dolfin
from fenics import *
import importlib
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
import physics.fluid_mechanics as flu
import function as fu
import function_spaces as fsp
import parameters.read.solution as rpam
import physics.utils as phys
import runtime_arguments as rarg
import switch_problem as swi
import variational_problem.utils as var_pr

import print_out_solution as pr_sol

# test transfer_mesh_to_sub_mesh - start 
import function as fu
import input_output as io
import mesh.load as lmsh
import runtime_arguments as rarg
import solution_paths as solpath



class u_0_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]**2-x[1]
        values[1] = x[0]**3-x[1]

    def value_shape(self):
        return (2,)

Q_0 = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 2)
Q_1 = VectorFunctionSpace(lmsh.sub_meshes[1], 'P', 2, dim=2)

u_0 = Function(Q_0)
u_1 = Function(Q_1)

u_0.interpolate(u_0_expression(element=Q_0.ufl_element()))


fu.transfer_mesh_to_sub_mesh(u_0, u_1, rarg.args.input_directory)

io.full_print(u_0, 'u_0_test', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
io.full_print(u_1, 'u_1_test', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)

# test transfer_mesh_to_sub_mesh - end

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size

# set the solver parameters here
# parameters with Netwon method
'''
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

# parameters with SNES method
# 
params = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'superlu',
        'line_search': 'bt',  # backtracking line search
        'absolute_tolerance': 1e-6,
        'relative_tolerance': 1e-6,
        'maximum_iterations': 1000000,
        'report': True,
    }
}

PETScOptions.clear()
PETScOptions.set('snes_type', 'newtontr')
PETScOptions.set('snes_atol', 1e-12)     # Stricter absolute tolerance
PETScOptions.set('snes_rtol', 1e-12)     # Stricter relative tolerance
PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 1000000)         # Increase function evaluation limit
# 


print(f'Generating initial mesh ...')

print(f'... done.')

rmsh = importlib.import_module(swi.rmsh)

# test calls of problems
# 1) membrane problem
fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rarg.args.input_directory)

vp_membrane = importlib.import_module(swi.vp_membrane)

# 2) mesh problem
# project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
# a) project U_n_12
v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh, rarg.args.input_directory)
# b) project U_dot_n_12
fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh, rarg.args.input_directory)

vp_mesh = importlib.import_module(swi.vp_mesh)


# 3) fluid problem
vp_fluid = importlib.import_module(swi.vp_fluid)


pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']

print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

fsp.sigma_n_32.interpolate( vp_membrane.sigma_n_32_0_Expression( element=fsp.Q_psi_n_12.ufl_element() ))


#Option 1: set initial profiles
# 1) for the membrane
fsp.v_bar_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_bar.ufl_element() ) )
fsp.v_n_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_n.ufl_element() ) )
fsp.nu_n_12_0.interpolate( vp_membrane.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ) )
fsp.U_n_12_0.interpolate( vp_membrane.U_n_12_0_Expression( element=fsp.Q_U_n_12.ufl_element() ) )
# 2) for the mesh
# 3) for the fluid
# fsp.v_n_1.interpolate(vp_fl.v_expression(element=fsp.Q_v.ufl_element()))
# fsp.v_n_2.assign(fsp.v_n_1)
fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)


#Option 2:read initial profiles by reading them from file


fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])


print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
for n in range(rpam.parameters['N']):
    # Update current time
    t += dt
    step += 1

    # step 1): update theta and omega
    print('Solving membrane problem ...', flush=True)
   

   
    # project from sub_mesh[0] onto sub_mesh[1] the fields from the fluid problem, in order to find the force exerted by the fluid on the membrane 
    fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
    fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rarg.args.input_directory)
    

    vp_membrane = importlib.import_module(swi.vp_membrane)

    var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)

    print('... done.', flush=True)



    # step 2): update u and u_dot (mesh problem)
    print('Solving mesh problem ...', flush=True)
    
    # project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
    # a) project U_n_12
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
    fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh, rarg.args.input_directory)
    # b) project U_dot_n_12
    fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh, rarg.args.input_directory)

    vp_mesh = importlib.reload(vp_mesh)

    # solve for u_n and u_dot_n
    var_pr.solve_vp(vp_mesh.F_msh, fsp.u_n, vp_mesh.bcs_msh, fsp.J_u, parameters=params)
    var_pr.solve_vp(vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot, fsp.J_u_dot, parameters=params)

    print('... done.', flush=True)

    

    # step 3: solve fluid problem
    print('Solving fluid problem ...', flush=True)

    vp_fluid = importlib.reload(vp_fluid)

    # step 3.1
    var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar, vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)

    # Step 3.2: surface_tension correction step
    var_pr.solve_vp(vp_fluid.F_phi_fl, fsp.phi_fl, vp_fluid.bc_phi_fl, fsp.J_phi_fl, parameters=params)

    # step 3.3
    var_pr.solve_vp(vp_fluid.F_v_fl_n, fsp.v_fl_n, [], fsp.J_v_fl_n, parameters=params)

    print('... done.', flush=True)
    


    pr_bc.print_bcs()

    
    # update the fields
    # 1) update the membrane problem 
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )

    fsp.v_n_2.assign( fsp.v_n_1 )
    fsp.v_n_1.assign( v_n_output )

    fsp.w_n_1.assign( w_n_output )

    fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
    fsp.sigma_n_32.assign( fsp.sigma_n_12 )

    fsp.U_n_32.assign( U_n_12_output )


    # 2) update the mesh problem
    fsp.u_n_2.assign(fsp.u_n_1)
    fsp.u_n_1.assign(fsp.u_n)

    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(fsp.u_dot_n)
    
    # 3) update the fluid problem
    fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)

    fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
    fsp.v_fl_n_1.assign(fsp.v_fl_n)

    fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    if step % rpam.parameters['print_out_stride'] == 0:
    # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        pr_sol.print_solution(t, step, dt)

    print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)
    
    

print("... done.", flush=True)
