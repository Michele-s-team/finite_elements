
import dolfin
from fenics import *
import importlib
import sys
import numpy as np
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

''' chaged the solver. nonlinear_newton more stable '''
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-5,
                  'relative_tolerance': 1e-5,
                  'maximum_iterations': 1000,
                  'relaxation_parameter': 0.95,
              }
          }
PETScOptions.clear()
# PETScOptions.set('snes_type', 'newtontr')  #fragile, switch to newtonls, more rebust. switch from trust-region to line-search snes

PETScOptions.set('snes_type', 'newtonls')       # line search instead of trust region
PETScOptions.set('snes_linesearch_type', 'bt')  # backtracking
PETScOptions.set('snes_linesearch_maxstep', '1.0')

PETScOptions.set('snes_atol', 1e-8)     # Stricter absolute tolerance (from 12 relax it to 8)
PETScOptions.set('snes_rtol', 1e-8)     # Stricter relative tolerance  (from 12 relax it to 8)
PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
PETScOptions.set('snes_max_it', 1000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 100000)         # Increase function evaluation limit
# 


rmsh = importlib.import_module(swi.rmsh)

# test calls of problems
# 1) membrane problem
fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])

vp_membrane = importlib.import_module(swi.vp_membrane)

# 2) mesh problem
# project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
# a) project U_n_12
v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
# b) project U_dot_n_12
fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)

fu.transfer_sub_mesh_to_mesh(u_fs_output,fsp.u_fs_on_mesh)
fu.transfer_sub_mesh_to_mesh(u_fs_dot_output,fsp.u_fs_dot_on_mesh)         

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


# print("Starting time iteration ...", flush=True)
# Time-stepping
t = 0
step = 0
n_bottom = Constant((0.0, -1.0))

for n in range(rpam.parameters['N']):
    # Update current time
    t += dt
    step += 1

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
  
    # MESH
    # Transfer interface displacement
    v_bar_output, w_bar_output, phi_output, \
    v_n_output, w_n_output, U_n_12_output, \
    nu_n_12_output, psi_n_12_output, mu_n_12_output = \
    fsp.psi_mem.split(deepcopy=True)

    fu.transfer_sub_mesh_to_mesh( U_n_12_output,fsp.U_n_12_on_mesh)

    fsp.U_dot_n_12.assign( project( phys.U_dot( fsp.w_n_1,   geo_al.normal( fsp.psi_n_12,   fsp.nu_n_12)),fsp.Q_U_dot_n_12))

    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12,fsp.U_dot_n_12_on_mesh)

    vp_mesh = importlib.reload(vp_mesh)

    var_pr.solve_vp(vp_mesh.F_msh,fsp.u_n,vp_mesh.bcs_msh,fsp.J_u,parameters=params,)

    var_pr.solve_vp( vp_mesh.F_msh_dot,fsp.u_dot_n,vp_mesh.bcs_msh_dot,fsp.J_u_dot,parameters=params,)


    # fsp.u_n.vector()[:] = (fsp.u_n.vector()[:]+ dt * fsp.u_dot_n.vector()[:])

    # step 1): update theta and omega

    # project from sub_mesh[0] onto sub_mesh[1] the fields from the fluid problem, in order to find the force exerted by the fluid on the membrane 
    fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n, fsp.sigma_fl_n_32, fsp.u_n, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
    fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])
    

    # vp_membrane = importlib.import_module(swi.vp_membrane) #iimport_module does not reload, it just returns the cached module. The variational forms are never updated with the new field, so use reload instead. 
    vp_membrane = importlib.reload(vp_membrane)

    var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)
    ''' the update: discrete form of the kinematic definition of velocity u(t+dt) = u(t) + \int(u_dot*ds). Approximating the integral with a 
     first oder Euler step gives: u^(n+1) = u^(n) + \delta t*u_dot^(n+1). Translated as velocity --> displacement.

     BUT this requires also another condition, thatof the balance of stress:
     ---kinematic equation:how surface moves
     ---dynamics condition: what forces act on the surface

     this is just enough for membrane moving implementation (membrane is being advected by velocity)
     '''

    pr_bc.print_bcs()   
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
    # fsp.u_n_1.assign(fsp.u_n)

    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(fsp.u_dot_n)
    
    # 3) update the fluid problem
    fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)


    fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
    fsp.v_fl_n_1.assign(fsp.v_fl_n)

    fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    fsp.U_bottom_2.assign(fsp.U_bottom_1)
    fsp.U_bottom_1.assign(fsp.u_fs_output)

    fsp.U_dot_bottom_2.assign(fsp.U_dot_bottom_1)
    fsp.U_dot_bottom_1.assign(fsp.u_fs_dot_output)

#free surface update
    fu.transfer_mesh_to_sub_mesh(
        fsp.u_n,
        fsp.u_fs_output,
        rmsh.parameters["h"]
    )

    # Advance it with the fluid velocity
    fsp.u_fs_output.vector().axpy(
        dt,
        fsp.u_fs_dot_output.vector()
    )
    fsp.assigner_fs.assign(fsp.psi_fs,[fsp.u_fs_output,fsp.u_fs_dot_output])

    fu.transfer_sub_mesh_to_mesh(fsp.u_fs_output,fsp.u_fs_on_mesh)

    fu.transfer_sub_mesh_to_mesh(fsp.u_fs_dot_output,fsp.u_fs_dot_on_mesh)

    #add
    if step % rpam.parameters['print_out_stride'] == 0:
    # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        pr_sol.print_solution(t, step, dt)

    print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)

    coords = rmsh.lmsh.sub_meshes[0].coordinates()

    print(f"t = {t:.6f}")
    print("||v|| =", fsp.v_fl_n.vector().norm("l2"))
    print("||u|| =", fsp.u_n.vector().norm("l2"))
    print("||u_dot|| =", fsp.u_dot_n.vector().norm("l2"))


print("||u_dot_n|| =", norm(fsp.u_n))
print("max u_dot =", fsp.u_n.vector().max())
print("min u_dot =", fsp.u_n.vector().min())




