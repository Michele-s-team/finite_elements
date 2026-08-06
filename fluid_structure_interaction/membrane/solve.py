
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
import numpy as np
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

''' chaged the solver. nonlinear_newton more stable '''
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-5,
                  'relative_tolerance': 1e-5,
                  'maximum_iterations': 1000,
                  'relaxation_parameter': 0.8,
              }
          }
PETScOptions.clear()

PETScOptions.set('snes_type', 'newtonls')       # line search instead of trust region
PETScOptions.set('snes_linesearch_type', 'bt')  # backtracking
PETScOptions.set('snes_linesearch_maxstep', '1.0')
PETScOptions.set('snes_atol', 1e-8)     # Stricter absolute tolerance (from 12 relax it to 8)
PETScOptions.set('snes_rtol', 1e-8)     # Stricter relative tolerance  (from 12 relax it to 8)
PETScOptions.set('snes_stol', 1e-8)    
PETScOptions.set('snes_max_it', 1000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 100000)    
rmsh = importlib.import_module(swi.rmsh)
# 3) fluid problem
vp_fluid = importlib.import_module(swi.vp_fluid)

pr_bc = importlib.import_module(swi.prout_bc)
dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']
print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)

# 1) membrane problem
fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n_1, fsp.sigma_fl_n_32, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])

vp_membrane = importlib.import_module(swi.vp_membrane)

fsp.sigma_n_32.interpolate(vp_membrane.sigma_n_32_0_Expression( element=fsp.Q_psi_n_12.ufl_element() ))


# 2) mesh problem
v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)

# b) project U_dot_n_12
fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)


vp_mesh = importlib.import_module(swi.vp_mesh)

# 1) for the membrane
fsp.v_bar_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_bar.ufl_element() ) )
fsp.v_n_0.interpolate( vp_membrane.v_n_0_Expression( element=fsp.Q_v_n.ufl_element() ) )
fsp.nu_n_12_0.interpolate( vp_membrane.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ) )
fsp.U_n_12_0.interpolate( vp_membrane.U_n_12_0_Expression( element=fsp.Q_U_n_12.ufl_element() ) )

fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_fl_n_12_Expression(element=fsp.Q_phi_fl.ufl_element()))
fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

fsp.assigner_mem.assign(fsp.psi_mem, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.U_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])




xdmffile_sigma_fl_tensor = XDMFFile(
    rarg.args.output_directory + "/sigma_fl_tensor.xdmf"
)
xdmffile_sigma_fl_tensor.parameters["flush_output"] = True
xdmffile_sigma_fl_tensor.parameters["functions_share_mesh"] = True
xdmffile_sigma_fl_tensor.parameters["rewrite_function_mesh"] = False


t = 0
step = 0
n_bottom = Constant((0.0, -1.0))


save_initial_frame = True

for n in range(rpam.parameters['N']):
    t += dt
    step += 1

    # --- fluid solve ---
    vp_fluid = importlib.reload(vp_fluid)
    var_pr.solve_vp(vp_fluid.F_v_fl_bar, fsp.v_fl_bar, vp_fluid.bc_v_fl_bar, fsp.J_v_fl_bar, parameters=params)
    var_pr.solve_vp(vp_fluid.F_phi_fl, fsp.phi_fl, [], fsp.J_phi_fl, parameters=params)
    var_pr.solve_vp(vp_fluid.F_v_fl_n, fsp.v_fl_n, [], fsp.J_v_fl_n, parameters=params)

    # Save the exact initial frame before any geometry update
    if save_initial_frame and step == 1:
        if step % rpam.parameters['print_out_stride'] == 0:
            pr_sol.print_solution(0.0, 0, dt)
        continue



    # --- membrane solve ---
    fsp.var_tensor_sigma_fl.assign(project(flu.sigma_ale(fsp.v_fl_n, fsp.sigma_fl_n_32, fsp.u_n, rpam.parameters['eta_fluid']),fsp.Q_var_tensor_sigma_fl))
    fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])

    if step % rpam.parameters['print_out_stride'] == 0:
        neg_sigma_fl_tensor = project( fsp.var_tensor_sigma_fl, fsp.Q_var_tensor_sigma_fl)
        xdmffile_sigma_fl_tensor.write(neg_sigma_fl_tensor, t)

    

    print('Solving membrane problem ...', flush=True)
    vp_membrane = importlib.reload(vp_membrane)
    var_pr.solve_vp(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, fsp.J_psi_mem, parameters=params)

    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split(deepcopy=True)

    # --- mesh solve ---
    fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
    fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)),fsp.Q_U_dot_n_12))
    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

    vp_mesh = importlib.reload(vp_mesh)
    var_pr.solve_vp(vp_mesh.F_msh, fsp.u_n, vp_mesh.bcs_msh, fsp.J_u, parameters=params)
    var_pr.solve_vp(vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot, fsp.J_u_dot, parameters=params)

    # Update mesh only after solving
    fsp.u_n.vector()[:] = fsp.u_n.vector()[:] + dt * fsp.u_dot_n.vector()[:]

    # Refresh histories after the update
    fsp.u_n_2.assign(fsp.u_n_1)
    fsp.u_n_1.assign(fsp.u_n)
    fsp.u_dot_n_2.assign(fsp.u_dot_n_1)
    fsp.u_dot_n_1.assign(fsp.u_dot_n)

    # --- update fluid  ---
    fsp.sigma_fl_n_12.assign(fsp.sigma_fl_n_32 - fsp.phi_fl)
    fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
    fsp.v_fl_n_1.assign(fsp.v_fl_n)
    fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)

    # --- update membrane histories ---
    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign(v_n_output)
    fsp.w_n_1.assign(w_n_output)
    fsp.sigma_n_12.assign(fsp.sigma_n_32 - project(phi_output, fsp.Q_phi))
    fsp.sigma_n_32.assign(fsp.sigma_n_12)
    fsp.U_n_32.assign(U_n_12_output)

    if step % rpam.parameters['print_out_stride'] == 0:
        pr_sol.print_solution(t, step, dt)
    print(f'\t{(100.0 * (t / rpam.parameters["T"]))} %', flush=True)

    print("||phi_fl|| =", norm(fsp.phi_fl))
    print("||sigma_fl_n_32|| =", norm(fsp.sigma_fl_n_32))
    print("||u_n|| =", norm(fsp.u_n))
    print("||u_dot_n|| =", norm(fsp.u_dot_n))


print("||u_dot_n|| =", norm(fsp.u_n))
print("max u_dot =", fsp.u_n.vector().max())
print("min u_dot =", fsp.u_n.vector().min())
u_vec = U_n_12_output.vector().get_local()