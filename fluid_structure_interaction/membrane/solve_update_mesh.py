#     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/membrane/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_line_a $MESH_PATH $SOLUTION_PATH
# """
# import dolfin
# from fenics import *
# import importlib
# import sys
# import numpy as np
# # add the path where to find the shared modules
# module_path = '/home/fenics/shared/modules'
# sys.path.append(module_path)

# import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
# import physics.fluid_mechanics as flu
# import function as fu
# import function_spaces as fsp
# import parameters.read.solution as rpam
# import physics.utils as phys
# import runtime_arguments as rarg
# import switch_problem as swi
# import variational_problem.utils as var_pr

# import print_out_solution as pr_sol

# dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size

# # set the solver parameters here
# # parameters with Netwon method
# '''
# params = {'nonlinear_solver': 'newton',
#           'newton_solver':
#               {
#                   'linear_solver': 'superlu',
#                   'absolute_tolerance': 1e-6,
#                   'relative_tolerance': 1e-6,
#                   'maximum_iterations': 1000000,
#                   'relaxation_parameter': 0.95,
#               }
#           }
# '''

# # parameters with SNES method

# # params = {
# #     'nonlinear_solver': 'snes',
# #     'snes_solver': {
# #         'linear_solver': 'mumps',
# #         'line_search': 'bt',  # backtracking line search
# #         'absolute_tolerance': 1e-6,
# #         'relative_tolerance': 1e-6,
# #         'maximum_iterations': 1000000,
# #         'report': True,
# #     }
# # }

# ''' chaged the solver. nonlinear_newton more stable '''
# params = {'nonlinear_solver': 'newton',
#           'newton_solver':
#               {
#                   'linear_solver': 'superlu',
#                   'absolute_tolerance': 1e-5,
#                   'relative_tolerance': 1e-5,
#                   'maximum_iterations': 1000,
#                   'relaxation_parameter': 0.95,
#               }
#           }
# PETScOptions.clear()
# # PETScOptions.set('snes_type', 'newtontr')  #fragile, switch to newtonls, more rebust. switch from trust-region to line-search snes

# PETScOptions.set('snes_type', 'newtonls')       # line search instead of trust region
# PETScOptions.set('snes_linesearch_type', 'bt')  # backtracking
# PETScOptions.set('snes_linesearch_maxstep', '1.0')

# PETScOptions.set('snes_atol', 1e-8)     # Stricter absolute tolerance (from 12 relax it to 8)
# PETScOptions.set('snes_rtol', 1e-8)     # Stricter relative tolerance  (from 12 relax it to 8)
# PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
# PETScOptions.set('snes_max_it', 1000)
# PETScOptions.set('snes_monitor')
# PETScOptions.set('snes_max_funcs', 100000)         # Increase function evaluation limit
# # 


# rmsh = importlib.import_module(swi.rmsh)

# # test calls of problems
# # 1) membrane problem
# fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
# fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])

# vp_membrane = importlib.import_module(swi.vp_membrane)

# # 2) mesh problem
# # project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
# # a) project U_n_12
# v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
# fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
# # b) project U_dot_n_12
# fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
# fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)


# ''' additional projections for the free boundary. free surface information now can be used by the buld equations beause we allow projection from sub_mesh_2
# into the sub_mesh_0. '''
# u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)

# fu.transfer_sub_mesh_to_mesh(u_fs_output,fsp.u_fs_on_mesh)
# fu.transfer_sub_mesh_to_mesh(u_fs_dot_output,fsp.u_fs_dot_on_mesh)         

# vp_mesh = importlib.import_module(swi.vp_mesh)


# # 3) fluid problem
# vp_fluid = importlib.import_module(swi.vp_fluid)


# pr_bc = importlib.import_module(swi.prout_bc)

# dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']

# print("Input directory", rarg.args.input_directory)
# print("Output directory", rarg.args.output_directory)

# fsp.sigma_n_32.interpolate( vp_membrane.sigma_n_32_0_Expression( element=fsp.Q_psi_n_12.ufl_element() ))


# #Option 1: set initial profiles
# # 1) for the membrane
# fsp.v_bar_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_bar.ufl_element() ) )
# fsp.v_n_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_n.ufl_element() ) )
# fsp.nu_n_12_0.interpolate( vp_membrane.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ) )
# fsp.U_n_12_0.interpolate( vp_membrane.U_n_12_0_Expression( element=fsp.Q_U_n_12.ufl_element() ) )
# # 2) for the mesh
# # 3) for the fluid
# # fsp.v_n_1.interpolate(vp_fl.v_expression(element=fsp.Q_v.ufl_element()))
# # fsp.v_n_2.assign(fsp.v_n_1)
# fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
# fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)


# #Option 2:read initial profiles by reading them from file


# fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])


# # print("Starting time iteration ...", flush=True)
# # Time-stepping
# t = 0
# step = 0
# n_bottom = Constant((0.0, -1.0))


# for n in range(rpam.parameters['N']):
#     # Update current time
#     t += dt
#     step += 1



#     # step 3: solve fluid problem
#     print('Solving fluid problem ...', flush=True)

#     vp_fluid = importlib.reload(vp_fluid)

#     # step 3.1
#     var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar, vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)

#     # Step 3.2: surface_tension correction step
#     var_pr.solve_vp(vp_fluid.F_phi_fl, fsp.phi_fl, vp_fluid.bc_phi_fl, fsp.J_phi_fl, parameters=params)

#     # step 3.3
#     var_pr.solve_vp(vp_fluid.F_v_fl_n, fsp.v_fl_n, [], fsp.J_v_fl_n, parameters=params)

#     print('... done.', flush=True)
    



#     # step 1): update theta and omega
#     # print('Solving membrane problem ...', flush=True)
   

   
#     # project from sub_mesh[0] onto sub_mesh[1] the fields from the fluid problem, in order to find the force exerted by the fluid on the membrane 
#     fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
#     fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])
    

#     # vp_membrane = importlib.import_module(swi.vp_membrane) #iimport_module does not reload, it just returns the cached module. The variational forms are never updated with the new field, so use reload instead. 
#     vp_membrane = importlib.reload(vp_membrane)

#     var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)
#     ''' the update: discrete form of the kinematic definition of velocity u(t+dt) = u(t) + \int(u_dot*ds). Approximating the integral with a 
#      first oder Euler step gives: u^(n+1) = u^(n) + \delta t*u_dot^(n+1). Translated as velocity --> displacement.

#      BUT this requires also another condition, thatof the balance of stress:
#      ---kinematic equation:how surface moves
#      ---dynamics condition: what forces act on the surface

#      this is just enough for membrane moving implementation (membrane is being advected by velocity)
#      '''


#     pr_bc.print_bcs()


#     fsp.u_n_2.assign(fsp.u_n_1)
#     fsp.u_n_1.assign(fsp.u_n)

#     fsp.u_n.vector()[:] = (fsp.u_n_1.vector()[:]+ dt * fsp.u_dot_n.vector()[:])
   
#     print('... done.', flush=True)



#     # step 2): update u and u_dot (mesh problem)
#     print('Solving mesh problem ...', flush=True)
    
#     # project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
#     # a) project U_n_12
#     v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
#     fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
#     # b) project U_dot_n_12
#     fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
#     fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

#     vp_mesh = importlib.reload(vp_mesh)

#     # solve for u_n and u_dot_n
#     var_pr.solve_vp(vp_mesh.F_msh, fsp.u_n, vp_mesh.bcs_msh, fsp.J_u, parameters=params)
#     var_pr.solve_vp(vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot, fsp.J_u_dot, parameters=params)

#     print('... done.', flush=True)

#     new_v_fl = project(fsp.v_fl_n, fsp.Q_u_dot)
#     fsp.u_fs_on_mesh.vector()[:] = (fsp.u_fs_on_mesh.vector()[:]+ dt * new_v_fl.vector()[:])
     


#     pr_bc.print_bcs()

    
#     # update the fields
#     # 1) update the membrane problem 
#     v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )

    
#     print("U min =", U_n_12_output.vector().min())
#     print("U max =", U_n_12_output.vector().max())

#     print("w min =", w_n_output.vector().min())
#     print("w max =", w_n_output.vector().max())


#     fsp.v_n_2.assign( fsp.v_n_1 )
#     fsp.v_n_1.assign( v_n_output )

#     fsp.w_n_1.assign( w_n_output )

#     fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
#     fsp.sigma_n_32.assign( fsp.sigma_n_12 )

#     fsp.U_n_32.assign( U_n_12_output )


#     # 2) update the mesh problem
#     fsp.u_n_2.assign(fsp.u_n_1)
#     fsp.u_n_1.assign(fsp.u_n)

#     fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
#     fsp.u_dot_n_1.assign(fsp.u_dot_n)
    
#     # 3) update the fluid problem
#     fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)

#     fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
#     fsp.v_fl_n_1.assign(fsp.v_fl_n)

#     fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    

#     #add

#     if step % rpam.parameters['print_out_stride'] == 0:
#     # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
#         pr_sol.print_solution(t, step, dt)

#     print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)

#     coords = rmsh.lmsh.sub_meshes[0].coordinates()
#     print(f"step {step}")
#     print("||u_n||      =", norm(fsp.u_n))
#     print("||u_n_1||    =", norm(fsp.u_n_1))
#     print("||u_dot_n||  =", norm(fsp.u_dot_n))



# u_vec = U_n_12_output.vector().get_local()















# """
# This code solves for the dynamics of the Navier Stokes equations for a fluid in a square whose top edge is a membrane. The coupled dynamics of  membrane, fluid and of the fictitious elastic body (which defines the region where the fluid moves) are solved. 

# run with:
#     rm -r solution; mkdir solution; python3 solve.py [path where to read the mesh] [path where to store the solution]

# Examples:
#     MESH_PATH="/home/fenics/shared/generate_mesh/2d/square_no_circle/line/solution"; SOLUTION_PATH="/home/fenics/shared/fluid_structure_interaction/membrane/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_no_circle_line_a $MESH_PATH $SOLUTION_PATH
# """
# import dolfin
# from fenics import *
# import importlib
# import sys
# import numpy as np
# # add the path where to find the shared modules
# module_path = '/home/fenics/shared/modules'
# sys.path.append(module_path)

# import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
# import physics.fluid_mechanics as flu
# import function as fu
# import function_spaces as fsp
# import parameters.read.solution as rpam
# import physics.utils as phys
# import runtime_arguments as rarg
# import switch_problem as swi
# import variational_problem.utils as var_pr

# import print_out_solution as pr_sol

# dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size

# # set the solver parameters here
# # parameters with Netwon method
# '''
# params = {'nonlinear_solver': 'newton',
#           'newton_solver':
#               {
#                   'linear_solver': 'superlu',
#                   'absolute_tolerance': 1e-6,
#                   'relative_tolerance': 1e-6,
#                   'maximum_iterations': 1000000,
#                   'relaxation_parameter': 0.95,
#               }
#           }
# '''

# # parameters with SNES method

# # params = {
# #     'nonlinear_solver': 'snes',
# #     'snes_solver': {
# #         'linear_solver': 'mumps',
# #         'line_search': 'bt',  # backtracking line search
# #         'absolute_tolerance': 1e-6,
# #         'relative_tolerance': 1e-6,
# #         'maximum_iterations': 1000000,
# #         'report': True,
# #     }
# # }

# ''' chaged the solver. nonlinear_newton more stable '''
# params = {'nonlinear_solver': 'newton',
#           'newton_solver':
#               {
#                   'linear_solver': 'superlu',
#                   'absolute_tolerance': 1e-5,
#                   'relative_tolerance': 1e-5,
#                   'maximum_iterations': 1000,
#                   'relaxation_parameter': 0.95,
#               }
#           }
# PETScOptions.clear()
# # PETScOptions.set('snes_type', 'newtontr')  #fragile, switch to newtonls, more rebust. switch from trust-region to line-search snes

# PETScOptions.set('snes_type', 'newtonls')       # line search instead of trust region
# PETScOptions.set('snes_linesearch_type', 'bt')  # backtracking
# PETScOptions.set('snes_linesearch_maxstep', '1.0')

# PETScOptions.set('snes_atol', 1e-8)     # Stricter absolute tolerance (from 12 relax it to 8)
# PETScOptions.set('snes_rtol', 1e-8)     # Stricter relative tolerance  (from 12 relax it to 8)
# PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
# PETScOptions.set('snes_max_it', 1000)
# PETScOptions.set('snes_monitor')
# PETScOptions.set('snes_max_funcs', 100000)         # Increase function evaluation limit
# # 


# rmsh = importlib.import_module(swi.rmsh)

# # test calls of problems
# # 1) membrane problem
# fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
# fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])

# vp_membrane = importlib.import_module(swi.vp_membrane)

# # 2) mesh problem
# # project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
# # a) project U_n_12
# v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
# fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
# # b) project U_dot_n_12
# fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
# fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)


# ''' additional projections for the free boundary. free surface information now can be used by the buld equations beause we allow projection from sub_mesh_2
# into the sub_mesh_0. '''
# u_fs_output, u_fs_dot_output = fsp.psi_fs.split(deepcopy=True)

# fu.transfer_sub_mesh_to_mesh(u_fs_output,fsp.u_fs_on_mesh)
# fu.transfer_sub_mesh_to_mesh(u_fs_dot_output,fsp.u_fs_dot_on_mesh)         

# vp_mesh = importlib.import_module(swi.vp_mesh)


# # 3) fluid problem
# vp_fluid = importlib.import_module(swi.vp_fluid)


# pr_bc = importlib.import_module(swi.prout_bc)

# dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']

# print("Input directory", rarg.args.input_directory)
# print("Output directory", rarg.args.output_directory)

# fsp.sigma_n_32.interpolate( vp_membrane.sigma_n_32_0_Expression( element=fsp.Q_psi_n_12.ufl_element() ))


# #Option 1: set initial profiles
# # 1) for the membrane
# fsp.v_bar_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_bar.ufl_element() ) )
# fsp.v_n_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_n.ufl_element() ) )
# fsp.nu_n_12_0.interpolate( vp_membrane.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ) )
# fsp.U_n_12_0.interpolate( vp_membrane.U_n_12_0_Expression( element=fsp.Q_U_n_12.ufl_element() ) )
# # 2) for the mesh
# # 3) for the fluid
# # fsp.v_n_1.interpolate(vp_fl.v_expression(element=fsp.Q_v.ufl_element()))
# # fsp.v_n_2.assign(fsp.v_n_1)
# fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
# fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)


# #Option 2:read initial profiles by reading them from file


# fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])


# # print("Starting time iteration ...", flush=True)
# # Time-stepping
# t = 0
# step = 0
# n_bottom = Constant((0.0, -1.0))


# for n in range(rpam.parameters['N']):
#     # Update current time
#     t += dt
#     step += 1



#     # step 3: solve fluid problem
#     print('Solving fluid problem ...', flush=True)

#     vp_fluid = importlib.reload(vp_fluid)

#     # step 3.1
#     var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar, vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)
#     # Step 3.2: surface_tension correction step
#     var_pr.solve_vp(vp_fluid.F_phi_fl, fsp.phi_fl, vp_fluid.bc_phi_fl, fsp.J_phi_fl, parameters=params)

#     # step 3.3
#     var_pr.solve_vp(vp_fluid.F_v_fl_n, fsp.v_fl_n, [], fsp.J_v_fl_n, parameters=params)
#     phi = fsp.phi_fl.vector().get_local()

#     print("phi min =", phi.min())
#     print("phi max =", phi.max())
#     print("phi norm =", np.linalg.norm(phi))
#     print('... done.', flush=True)
    




#     # ==========================================================
#     # MESH
#     # ==========================================================

#     # Transfer interface displacement
#     v_bar_output, w_bar_output, phi_output, \
#     v_n_output, w_n_output, U_n_12_output, \
#     nu_n_12_output, psi_n_12_output, mu_n_12_output = \
#     fsp.psi_mem.split(deepcopy=True)

#     fu.transfer_sub_mesh_to_mesh( U_n_12_output,fsp.U_n_12_on_mesh)

#     fsp.U_dot_n_12.assign( project( phys.U_dot( fsp.w_n_1,   geo_al.normal( fsp.psi_n_12,   fsp.nu_n_12)),fsp.Q_U_dot_n_12))

#     fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12,fsp.U_dot_n_12_on_mesh)

#     vp_mesh = importlib.reload(vp_mesh)

#     var_pr.solve_vp(vp_mesh.F_msh,fsp.u_n,vp_mesh.bcs_msh,fsp.J_u,parameters=params,)

#     var_pr.solve_vp( vp_mesh.F_msh_dot,fsp.u_dot_n,vp_mesh.bcs_msh_dot,fsp.J_u_dot,parameters=params,)

#     new_v_fl = project(fsp.v_fl_n,fsp.Q_u_dot)

#     fsp.u_n.vector()[:] = (fsp.u_n.vector()[:]+ dt * fsp.u_dot_n_1.vector()[:])

#     # step 1): update theta and omega
#     # print('Solving membrane problem ...', flush=True)
   

   
#     # project from sub_mesh[0] onto sub_mesh[1] the fields from the fluid problem, in order to find the force exerted by the fluid on the membrane 
#     fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n, fsp.sigma_fl_n_32, fsp.u_n, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
#     fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])
    

#     # vp_membrane = importlib.import_module(swi.vp_membrane) #iimport_module does not reload, it just returns the cached module. The variational forms are never updated with the new field, so use reload instead. 
#     vp_membrane = importlib.reload(vp_membrane)

#     var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)
#     ''' the update: discrete form of the kinematic definition of velocity u(t+dt) = u(t) + \int(u_dot*ds). Approximating the integral with a 
#      first oder Euler step gives: u^(n+1) = u^(n) + \delta t*u_dot^(n+1). Translated as velocity --> displacement.

#      BUT this requires also another condition, thatof the balance of stress:
#      ---kinematic equation:how surface moves
#      ---dynamics condition: what forces act on the surface

#      this is just enough for membrane moving implementation (membrane is being advected by velocity)
#      '''


#     pr_bc.print_bcs()


   
#     print('... done.', flush=True)




#     pr_bc.print_bcs()

    
#     # update the fields
#     # 1) update the membrane problem 
#     v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )

#     fsp.v_n_2.assign( fsp.v_n_1 )
#     fsp.v_n_1.assign( v_n_output )

#     fsp.w_n_1.assign( w_n_output )

#     fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
#     fsp.sigma_n_32.assign( fsp.sigma_n_12 )

#     fsp.U_n_32.assign( U_n_12_output )


#     # 2) update the mesh problem
#     fsp.u_n_2.assign(fsp.u_n_1)
#     # fsp.u_n_1.assign(fsp.u_n)

#     fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
#     fsp.u_dot_n_1.assign(fsp.u_dot_n)
    
#     # 3) update the fluid problem
#     fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)


#     fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
#     fsp.v_fl_n_1.assign(fsp.v_fl_n)

#     fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    

#     #add

#     if step % rpam.parameters['print_out_stride'] == 0:
#     # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
#         pr_sol.print_solution(t, step, dt)

#     print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)

#     coords = rmsh.lmsh.sub_meshes[0].coordinates()

#     print(f"t = {t:.6f}")
#     print("||v|| =", fsp.v_fl_n.vector().norm("l2"))
#     print("||u|| =", fsp.u_n.vector().norm("l2"))
#     print("||u_dot|| =", fsp.u_dot_n.vector().norm("l2"))



# u_vec = U_n_12_output.vector().get_local()




# import dolfin
# from fenics import *
# import importlib
# import sys
# import numpy as np
# # add the path where to find the shared modules
# module_path = '/home/fenics/shared/modules'
# sys.path.append(module_path)

# import differential_geometry.manifold.gauges.arc_length_gauge as geo_al
# import physics.fluid_mechanics as flu
# import function as fu
# import function_spaces as fsp
# import parameters.read.solution as rpam
# import physics.utils as phys
# import runtime_arguments as rarg
# import switch_problem as swi
# import variational_problem.utils as var_pr

# import print_out_solution as pr_sol

# dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size
# # Fixed time-step size, computed once from the total simulation time T and number of steps N.

# # set the solver parameters here
# # parameters with Newton method
# ''' chaged the solver. nonlinear_newton more stable '''
# params = {'nonlinear_solver': 'newton',
#           'newton_solver':
#               {
#                   'linear_solver': 'superlu',
#                   'absolute_tolerance': 1e-5,
#                   'relative_tolerance': 1e-5,
#                   'maximum_iterations': 1000,
#                   'relaxation_parameter': 0.8,
#               }
#           }
# # Dictionary of Newton-solver parameters passed to every nonlinear solve in the time loop:
# # - 'superlu' as the linear solver used inside each Newton iteration,
# # - loose-ish tolerances (1e-5) and a damped step (relaxation 0.95) to help convergence,
# # - up to 1000 Newton iterations allowed per nonlinear solve.

# PETScOptions.clear()
# # Wipes any previously set PETSc options, so the options set below start from a clean slate
# # (important if this script is re-run in the same process / notebook).

# PETScOptions.set('snes_type', 'newtonls')       # line search instead of trust region
# PETScOptions.set('snes_linesearch_type', 'bt')  # backtracking
# PETScOptions.set('snes_linesearch_maxstep', '1.0')
# # Configures PETSc's SNES (nonlinear solver) to use a line-search Newton method (newtonls)
# # with backtracking line search, capped at a full Newton step (maxstep = 1.0). This is used
# # as an alternative/underlying mechanism to the 'newton_solver' dict above, depending on
# # how var_pr.solve_vp is implemented (it may route through PETSc SNES).

# PETScOptions.set('snes_atol', 1e-8)     # Stricter absolute tolerance (from 12 relax it to 8)
# PETScOptions.set('snes_rtol', 1e-8)     # Stricter relative tolerance  (from 12 relax it to 8)
# PETScOptions.set('snes_stol', 1e-8)      # Keep step tolerance same
# PETScOptions.set('snes_max_it', 1000)
# PETScOptions.set('snes_monitor')
# PETScOptions.set('snes_max_funcs', 100000)         # Increase function evaluation limit
# # Sets the actual PETSc SNES convergence tolerances/limits: absolute, relative, and step
# # tolerances all at 1e-8 (tighter than the 'newton_solver' dict above), a hard cap of 1000
# # nonlinear iterations, verbose per-iteration monitoring ('snes_monitor'), and a generous
# # ceiling on the number of residual/function evaluations SNES is allowed to perform.

# rmsh = importlib.import_module(swi.rmsh)
# # Dynamically imports the mesh module named in switch_problem.py (swi.rmsh). This module
# # defines the geometry: the full mesh, its sub-meshes (fluid domain, membrane line, etc.),
# # facet markers, and derived measures (dx, ds) used by all three variational-problem modules.

# # test calls of problems


# # 3) fluid problem
# vp_fluid = importlib.import_module(swi.vp_fluid)
# # Imports the fluid variational-problem module (defines F_v_fl_bar, F_phi_fl, F_v_fl_n, BCs).
# # Import-time interpolates the inlet velocity profile v_fl_bar_b and builds its DirichletBCs.

# pr_bc = importlib.import_module(swi.prout_bc)
# # Imports a small utility module used purely for debugging/printing the current boundary
# # conditions (pr_bc.print_bcs()), called inside the time loop below.

# dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']
# # Sets the polynomial degree used by FEniCS's form compiler (FFC) when generating quadrature
# # rules for all variational forms -- controls numerical integration accuracy vs. compile/solve cost.

# print("Input directory", rarg.args.input_directory)
# print("Output directory", rarg.args.output_directory)
# # Diagnostic prints confirming where input parameters were read from and where output will
# # be written, based on command-line arguments parsed in runtime_arguments.py.



# # 1) membrane problem
# fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
# fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])
# # INITIALIZATION ONLY (runs once, before the time loop):
# # Computes the fluid's ALE stress tensor sigma_ale(...) using the fluid velocity, pressure-like
# # field, and mesh displacement all at their initial ("n_1") values, projects it onto the full
# # mesh's tensor function space (Q_var_tensor_sigma_fl), then transfers that field from the full
# # mesh onto the membrane's sub-mesh (var_tensor_sigma_fl_on_mem). This gives the membrane problem
# # an initial guess for the force the fluid exerts on it, before any time-stepping has occurred.

# vp_membrane = importlib.import_module(swi.vp_membrane)
# # Imports the membrane variational-problem module (defines F_mem, bcs_mem, etc.) and, as a
# # side effect of import, builds the initial boundary conditions and initial-condition
# # expressions defined inside that module (e.g. fsp.X_ref, fsp.v_bar_l/r interpolation).


# fsp.sigma_n_32.interpolate(vp_membrane.sigma_n_32_0_Expression( element=fsp.Q_psi_n_12.ufl_element() ))
# # Sets the initial membrane tension/stress field sigma_n_32 to a constant value defined in
# # vp_membrane.sigma_n_32_0_Expression (read from rpam.parameters['sigma_n_12_0']).



# # 2) mesh problem
# # project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
# # a) project U_n_12
# v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
# fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
# # INITIALIZATION ONLY: splits the mixed membrane function psi_mem into its 9 individual
# # component fields (deepcopy=True makes independent Function objects, not views), then takes
# # the membrane displacement U_n_12_output and transfers it from the membrane's sub-mesh onto
# # the full mesh (fsp.U_n_12_on_mesh), so the mesh problem can eventually use it as a boundary
# # condition (currently unused directly in the mesh BCs -- see note below).

# # b) project U_dot_n_12
# fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
# fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)
# # INITIALIZATION ONLY: computes the membrane's normal displacement velocity U_dot from the
# # normal velocity component w_n_1 and the geometric surface normal, projects it onto the
# # membrane sub-mesh function space, then transfers it onto the full mesh
# # (fsp.U_dot_n_12_on_mesh), again for potential use as a mesh-problem BC.

# vp_mesh = importlib.import_module(swi.vp_mesh)
# # Imports the mesh variational-problem module (defines F_msh, F_msh_dot, bcs_msh, bcs_msh_dot).
# # Import-time also builds the DirichletBC objects for u and u_dot (fixed top, free-slip sides,
# # bottom BC tied to -v_fl_n).

# #Option 1: set initial profiles
# # 1) for the membrane
# fsp.v_bar_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_bar.ufl_element() ) )
# fsp.v_n_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_n.ufl_element() ) )
# fsp.nu_n_12_0.interpolate( vp_membrane.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ) )
# fsp.U_n_12_0.interpolate( vp_membrane.U_n_12_0_Expression( element=fsp.Q_U_n_12.ufl_element() ) )
# # Initializes the membrane's initial conditions at t=0: tangential velocity (v_bar_0, v_n_0)
# # set to the constant rpam.parameters['v_bar_l'][0], the "nu" gauge variable nu_n_12_0 set to 1
# # everywhere (arc-length gauge normalization), and the initial displacement U_n_12_0 set to
# # zero (membrane starts undisplaced from its reference straight-line configuration).

# # 2) for the mesh
# # 3) for the fluid
# # fsp.v_n_1.interpolate(vp_fl.v_expression(element=fsp.Q_v.ufl_element()))
# # fsp.v_n_2.assign(fsp.v_n_1)
# fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
# fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)
# # No explicit initial velocity is set for the fluid here (those lines are commented out, so
# # fluid velocity fields start at their FEniCS default, typically zero). The fluid's initial
# # pressure-like field sigma_fl_n_12 is set to a constant (rpam.parameters['sigma_fl_n_12_0_b']),
# # and sigma_fl_n_32 (used one half-step ahead in the scheme) is initialized equal to it.

# #Option 2: read initial profiles by reading them from file
# # (Not used here -- Option 1 above sets everything analytically instead of from a restart file.)

# fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])
# # Packs all 9 individually-initialized membrane sub-fields into the single mixed function
# # psi_mem, using a FunctionAssigner (fsp.assigner_mem). This is what the membrane's nonlinear
# # solve will use/update as its unknown each time step.

# # print("Starting time iteration ...", flush=True)
# # Time-stepping
# t = 0
# step = 0
# n_bottom = Constant((0.0, -1.0))
# # Initializes the simulation clock (t), step counter (step), and a constant downward unit
# # vector n_bottom = (0, -1) -- presumably intended as the outward normal on the bottom
# # boundary for use somewhere in the solve (currently unused in the loop body shown).

# for n in range(rpam.parameters['N']):
#     # Update current time
#     t += dt
#     step += 1
#     # Advances simulation time by one step and increments the step counter. rpam.parameters['N']
#     # is the total number of time steps, so this loop runs the whole simulation.

#     # =========================================================
#     # STEP 1: FLUID PROBLEM (solved first)
#     # =========================================================
#     print('Solving fluid problem ...', flush=True)

#     vp_fluid = importlib.reload(vp_fluid)
#     # Reloads the fluid variational-problem module fresh each time step. This re-executes
#     # the module's top-level code, which rebuilds the UFL forms F_v_fl_bar/F_phi_fl/F_v_fl_n
#     # using the CURRENT values of fsp.v_fl_n_1, fsp.u_n_1, etc. (necessary because UFL forms
#     # capture Python-level references, and some quantities inside vp_fluid are only refreshed
#     # by re-running the module, not automatically).

#     # step 1.1
#     var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar, vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)
#     # Solves the momentum-prediction step: an intermediate ("starred"/bar) fluid velocity
#     # v_fl_bar, using a BDF2-like time discretization and viscous+pressure terms from the
#     # previous time level. This is the first stage of the projection (Chorin-type) scheme.

#     # Step 1.2: surface_tension correction step
#     var_pr.solve_vp(vp_fluid.F_phi_fl, fsp.phi_fl, vp_fluid.bc_phi_fl, fsp.J_phi_fl, parameters=params)
#     # Solves a Poisson-type equation for the pressure-correction field phi_fl, enforcing
#     # (approximate) incompressibility of the corrected velocity field.

#     # step 1.3
#     var_pr.solve_vp(vp_fluid.F_v_fl_n, fsp.v_fl_n, [], fsp.J_v_fl_n, parameters=params)
#     # Corrects the intermediate velocity v_fl_bar using the gradient of phi_fl to produce the
#     # final, (approximately) divergence-free velocity v_fl_n for this time step. No Dirichlet
#     # BCs are imposed here ([]) since the correction step doesn't need its own boundary data.




#     print('... done.', flush=True)
#     # End of the fluid solve for this time step: v_fl_n now holds the current fluid velocity.

#     # =========================================================
#     # STEP 2: MEMBRANE PROBLEM (solved second, using the fresh fluid solution)
#     # =========================================================

#     # project from sub_mesh[0] onto sub_mesh[1] the fields from the fluid problem, in order to find the force exerted by the fluid on the membrane
#     fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n, fsp.sigma_fl_n_32, fsp.u_n, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
#     fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])
#     # Computes the fluid's ALE Cauchy-type stress tensor using the JUST-SOLVED fluid velocity
#     # v_fl_n, the current pressure-like field sigma_fl_n_32, and the mesh displacement u_n
#     # from the END of the PREVIOUS time step (this step's mesh hasn't been solved yet -- that
#     # now happens after the membrane, per the requested reordering). Projects the stress onto
#     # the full mesh, then transfers it onto the membrane's sub-mesh (var_tensor_sigma_fl_on_mem)
#     # so the membrane's variational form can use it as the force the fluid exerts on it.

#     print('Solving membrane problem ...', flush=True)

#     vp_membrane = importlib.reload(vp_membrane)
#     # Reloads the membrane module so its variational form F_mem is rebuilt using the freshly
#     # updated var_tensor_sigma_fl_on_mem (and any other current field values it references).
#     # (import_module would return the cached module and NOT pick up the new field -- reload
#     # is required here.)

#     var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)
#     # Solves the full nonlinear mixed membrane problem for psi_mem (all 9 sub-fields at once:
#     # v_bar, w_bar, phi, v_n, w_n, U_n_12, nu_n_12, psi_n_12, mu_n_12), using the membrane's
#     # own Dirichlet BCs (bcs_mem) and the fluid force just projected onto it. This determines
#     # how the membrane moves/deforms and what internal tension it develops this step.

#     print('... done.', flush=True)

#     # split the freshly solved membrane fields so they can be used below (mesh problem + updates)
#     v_bar_output, w_bar_output, phi_output, \
#     v_n_output, w_n_output, U_n_12_output, \
#     nu_n_12_output, psi_n_12_output, mu_n_12_output = \
#     fsp.psi_mem.split(deepcopy=True)
#     # Splits the just-updated mixed function psi_mem into its 9 independent component
#     # Functions again, now holding this time step's solved values (not last step's).

#     # =========================================================
#     # STEP 3: MESH PROBLEM (solved third, using the fresh membrane solution)
#     # =========================================================

#     fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
#     # Transfers the membrane's FRESHLY solved displacement field U_n_12_output from the
#     # membrane sub-mesh onto the full mesh (fsp.U_n_12_on_mesh). Because the membrane was
#     # solved before the mesh this iteration, this data is now up to date for this time step
#     # (previously this transfer used last step's membrane solution).

#     fsp.U_dot_n_12.assign( project( phys.U_dot( fsp.w_n_1, geo_al.normal( fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
#     fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)
#     # Computes the membrane's normal displacement velocity from the (previous) normal velocity
#     # w_n_1 and the current surface normal, projects it onto the membrane's function space,
#     # then transfers it onto the full mesh as fsp.U_dot_n_12_on_mesh, again now using this
#     # step's freshly updated geometry (psi_n_12, nu_n_12).

#     vp_mesh = importlib.reload(vp_mesh)
#     # Reloads the mesh variational-problem module so its forms F_msh/F_msh_dot are rebuilt
#     # against the current field values (e.g. u_n from the end of the previous step, and the
#     # bottom-boundary BC bc_u_dot_b which depends on the just-solved fsp.v_fl_n).

#     var_pr.solve_vp(vp_mesh.F_msh, fsp.u_n, vp_mesh.bcs_msh, fsp.J_u, parameters=params,)
#     # Solves the (hyper)elastic mesh-displacement problem for u_n: an elastic-type equation
#     # that smoothly extends the boundary displacement into the mesh interior, keeping the
#     # mesh well-shaped as the geometry evolves. Uses bcs_msh: zero x-displacement on the left/
#     # right edges, and zero displacement on the top edge (fixed).

#     var_pr.solve_vp( vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot, fsp.J_u_dot, parameters=params,)
#     # Solves for the mesh VELOCITY u_dot_n (linearized rate-of-change companion to F_msh),
#     # using bcs_msh_dot -- crucially including bc_u_dot_b = -v_fl_n on the bottom boundary
#     # (sub_mesh_2_id), which is what makes the bottom boundary move according to the fluid
#     # velocity that was just solved in Step 1. Top is fixed (zero), left/right are free-slip
#     # in x (zero x-component) to support periodicity.

#     fsp.u_n.vector()[:] = (fsp.u_n.vector()[:]+ dt * fsp.u_dot_n.vector()[:])
#     # Advances the mesh position: u_n <- u_n + dt * u_dot_n. This is the explicit Euler update
#     # that actually moves the bottom (and interior) mesh nodes forward according to the mesh
#     # velocity field -- since bc_u_dot_b ties u_dot_n on the bottom to -v_fl_n, this is the
#     # step where "the bottom boundary moves according to the fluid velocity" takes effect.

#     print("||u_dot_n_1|| =", norm(fsp.u_dot_n_1))
#     print("max u_dot_n_1 =", fsp.u_dot_n_1.vector().max())
#     print("min u_dot_n_1 =", fsp.u_dot_n_1.vector().min())




#     pr_bc.print_bcs()
#     print('... done.', flush=True)
#     # Debug print of the current boundary condition values/state, plus a completion message
#     # for the mesh solve.

#     # =========================================================
#     # UPDATE HISTORY VARIABLES for the next time step's BDF2-type discretization
#     # =========================================================
#     # 3) update the fluid problem
#     fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)
#     # Updates the fluid's pressure-like field at the half-step level, subtracting the just-
#     # solved correction phi_fl from the previous half-step value (mirrors the membrane's
#     # sigma update, but for the fluid).

#     fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
#     fsp.v_fl_n_1.assign(fsp.v_fl_n)
#     # Shifts the fluid velocity history back one level, then sets the new "n-1" value to this
#     # step's freshly solved v_fl_n -- feeds the BDF2-type convective terms in F_v_fl_bar next
#     # iteration.

#     fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)
#     # Shifts the fluid's pressure-like field forward to the new half-step value, ready for use
#     # in next iteration's fluid momentum equation.


#     # 1) update the membrane problem
#     fsp.v_n_2.assign( fsp.v_n_1 )
#     fsp.v_n_1.assign( v_n_output )
#     # Shifts the membrane's tangential-velocity history back one level: what was "n-1" becomes
#     # "n-2", and the freshly solved value becomes the new "n-1" -- needed for the two-level
#     # (BDF2-like) time-stepping scheme used in F_v_bar/F_v_fl_bar.

#     fsp.w_n_1.assign( w_n_output )
#     # Updates the membrane's normal velocity history (single-level here) to this step's value.

#     fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
#     fsp.sigma_n_32.assign( fsp.sigma_n_12 )
#     # Updates the membrane's tension/stress: subtracts the solved correction field phi_output
#     # (projected onto the scalar space Q_phi) from the previous half-step tension, producing
#     # the tension at the new half-step, then shifts it forward as the new "n+1/2" value for
#     # next iteration.

#     fsp.U_n_32.assign( U_n_12_output )
#     # Shifts the membrane displacement history forward: this step's solved U_n_12 becomes the
#     # "previous" value (U_n_32) that next iteration's F_U_n_12 will use as its baseline.

#     # 2) update the mesh problem
#     fsp.u_n_2.assign(fsp.u_n_1)
#     fsp.u_n_1.assign(fsp.u_n)
#     # Shifts the mesh-displacement history back one level (u_n_1 -> u_n_2). NOTE: the line
#     # that would set u_n_1 <- u_n (the freshly solved/advanced displacement) is commented out,
#     # so u_n_1 is never refreshed here -- worth checking whether this is intentional, since
#     # F_v_fl_bar/F_msh reference fsp.u_n_1 as "the current mesh configuration."

#     fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
#     fsp.u_dot_n_1.assign(fsp.u_dot_n)
#     # Shifts the mesh-velocity history back one level and then updates u_dot_n_1 with this
#     # step's freshly solved mesh velocity u_dot_n.




#     v_b = project(fsp.v_fl_n, fsp.Q_u_dot)
#     diff = Function(fsp.Q_u_dot)
#     diff.vector()[:] = fsp.u_dot_n.vector()[:] - v_b.vector()[:]

#     print("||u_dot - v|| =", norm(diff))


#     if step % rpam.parameters['print_out_stride'] == 0:
#         print("step", step)
#         print("||v_n_1-v_n|| =", norm(fsp.v_fl_n_1.vector()-fsp.v_fl_n.vector()))
#         print("||u_n_1-u_n|| =", norm(fsp.u_n_1.vector()-fsp.u_n.vector()))
#     # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
#         pr_sol.print_solution(t, step, dt)
#     # Every print_out_stride steps, writes the current solution fields to output files (e.g.
#     # for visualization in ParaView), avoiding writing output every single time step.

#     print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)
#     # Prints the simulation's percentage progress toward the total time T.

#     coords = rmsh.lmsh.sub_meshes[0].coordinates()
#     # Grabs the current coordinates of the fluid sub-mesh (sub_meshes[0]) -- computed but not
#     # used further in this snippet (likely leftover debugging code, or used implicitly by
#     # something not shown).

#     print(f"t = {t:.6f}")
#     print("||v|| =", fsp.v_fl_n.vector().norm("l2"))
#     print("||u|| =", fsp.u_n.vector().norm("l2"))
#     print("||u_dot|| =", fsp.u_dot_n.vector().norm("l2"))
#     # Diagnostic prints of the current time and the L2 norms of the fluid velocity, mesh
#     # displacement, and mesh velocity vectors -- useful for monitoring solution magnitude/
#     # divergence over the course of the simulation.


# print("||u_dot_n|| =", norm(fsp.u_n))
# print("max u_dot =", fsp.u_n.vector().max())
# print("min u_dot =", fsp.u_n.vector().min())
# # Final diagnostic prints after the time loop completes: the FEniCS built-in norm() of the
# # final mesh displacement field (note: variable name says "u_dot_n" but it's actually norm of
# # u_n, not u_dot_n -- likely a copy-paste labeling mistake), plus the max/min displacement
# # component values, giving a quick sanity check on the final mesh state.
# u_vec = U_n_12_output.vector().get_local()













