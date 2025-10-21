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
import elasticity as ela
import function as fu
import function_spaces as fsp
import parameters.read.solution as rpam
import physics as phys
import runtime_arguments as rarg
import switch_problem as swi

import print_out_solution as pr_sol

dt = rpam.parameters['T'] / rpam.parameters['N']  # time step size

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

rmsh = importlib.import_module(swi.rmsh)

# test calls of problems
# 1) membrane problem
fsp.var_tensor_sigma_fl.assign(project(ela.var_sigma_tensor(fsp.sigma_fl_n_32, fsp.v_fl_n_1, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
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

vp_mesh = importlib.import_module(swi.vp_mesh)


# 3) fluid problem
vp_fluid = importlib.import_module(swi.vp_fluid)


pr_bc = importlib.import_module(swi.prout_bc)

dolfin.parameters["form_compiler"]["quadrature_degree"] = rpam.parameters['quadrature_degree']

print("Input directory", rarg.args.input_directory)
print("Output directory", rarg.args.output_directory)


'''
# set the initial profiles:
# 1) for the membrane
# 2) for the fictitious elastic body 
# 3) for the fluid
fsp.v_fl_n_1.interpolate(vp_fluid.v_fl_0_Expression(element=fsp.Q_v_n.ufl_element()))
fsp.v_fl_n_2.assign(fsp.v_fl_n_1)
fsp.sigma_fl_n_12.interpolate(vp_fluid.sigma_expression(element=fsp.Q_phi_fl.ufl_element()))
fsp.sigma_fl_n_32.assign(fsp.sigma_fl_n_12)
'''



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
    fsp.var_tensor_sigma_fl.assign(project(ela.var_sigma_tensor(fsp.sigma_fl_n_32, fsp.v_fl_n_1, fsp.u_n_1, rpam.parameters['eta_fluid']), fsp.Q_var_tensor_sigma_fl))
    fu.transfer_mesh_to_sub_mesh(fsp.var_tensor_sigma_fl, fsp.var_tensor_sigma_fl_on_mem, rmsh.parameters['h'])
    
    # sign

    vp_membrane = importlib.import_module(swi.vp_membrane)
    
    J_mem = derivative(vp_membrane.F_mem, fsp.psi_mem, fsp.J_psi_mem)
    problem_mem = NonlinearVariationalProblem(vp_membrane.F_mem, fsp.psi_mem, vp_membrane.bcs_mem, J_mem)
    solver_mem = NonlinearVariationalSolver(problem_mem)
    solver_mem.parameters.update(params)
    solver_mem.solve()

    print('... done.', flush=True)

    '''
    # step 2): update u and u_dot (mesh problem)
    print('Solving mesh problem ...', flush=True)
    
    # project field U_n_12 and its time derivative from sub_mesh[0] onto sub_mesh[1] in order to set BCs for the mesh problem
    # a) project U_n_12
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, U_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi_mem.split( deepcopy=True )
    fu.transfer_sub_mesh_to_mesh(U_n_12_output, fsp.U_n_12_on_mesh)
    # b) project U_dot_n_12
    fsp.U_dot_n_12.assign(project(phys.U_dot(fsp.w_n_1, geo_al.normal(fsp.psi_n_12, fsp.nu_n_12)), fsp.Q_U_dot_n_12))
    fu.transfer_sub_mesh_to_mesh(fsp.U_dot_n_12, fsp.U_dot_n_12_on_mesh)

    vp_mesh = importlib.reload(vp_mesh)

    J_msh = derivative(vp_mesh.F_msh, fsp.u_n, fsp.J_u)
    problem_msh = NonlinearVariationalProblem(vp_mesh.F_msh, fsp.u_n, vp_mesh.bcs_msh, J_msh)
    solver_msh = NonlinearVariationalSolver(problem_msh)

    J_msh_dot = derivative(vp_mesh.F_msh_dot, fsp.u_dot_n, fsp.J_u_dot)
    problem_msh_dot = NonlinearVariationalProblem(vp_mesh.F_msh_dot, fsp.u_dot_n, vp_mesh.bcs_msh_dot, J_msh_dot)
    solver_msh_dot = NonlinearVariationalSolver(problem_msh_dot)

    solver_msh.parameters.update(params)
    solver_msh_dot.parameters.update(params)

    # solve for u and u_dot
    solver_msh.solve()
    solver_msh_dot.solve()

    print('... done.', flush=True)

    # step 3) update v_n and sigma_n_12 (fluid problem)
    print('Solving fluid problem ...', flush=True)

    vp_fluid = importlib.reload(vp_fluid)

    # step 3.1
    J_fluid_1 = derivative(vp_fluid.F_v_, fsp.v_, fsp.J_v_)
    problem_fluid_1 = NonlinearVariationalProblem(vp_fluid.F_v_, fsp.v_, vp_fluid.bc_v_fl_bar, J_fluid_1)
    solver_fluid_1 = NonlinearVariationalSolver(problem_fluid_1)
    solver_fluid_1.solve()

    # Step 3.2: surface_tension correction step
    J_fluid_2 = derivative(vp_fluid.F_phi, fsp.phi, fsp.J_phi)
    problem_fluid_2 = NonlinearVariationalProblem(vp_fluid.F_phi, fsp.phi, vp_fluid.bc_phi_fl, J_fluid_2)
    solver_fluid_2 = NonlinearVariationalSolver(problem_fluid_2)
    solver_fluid_2.solve()

    # step 3.3
    J_fluid_3 = derivative(vp_fluid.F_v_n, fsp.v_n, fsp.J_v_n)
    problem_fluid_3 = NonlinearVariationalProblem(vp_fluid.F_v_n, fsp.v_n, [], J_fluid_3)
    solver_fluid_3 = NonlinearVariationalSolver(problem_fluid_3)
    solver_fluid_3.solve()

    print('... done.', flush=True)

    pr_bc.print_bcs()

    # update the fields
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

    pr_sol.print_solution(t, step, dt)

    print("\t%.2f %%" % (100.0 * (t / rpam.T)), flush=True)
    
    '''

print("... done.", flush=True)
