'''
This file solves for the dynamics of a one-dimensional fluid whose reference configuration is a line, in the generalized arc-length gauge

This file needs the mesh files, which can be generated, for example, by `finite_elements/mesh/generate_mesh.py` with
python3 generate_mesh.py 0.1
and which are stored into finite_elements/mesh/solution

Run with
clear; clear; rm -rf solution; mkdir solution; python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]

Examples:

    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/lagrangian_approach/one_dimension/line/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_a $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/lagrangian_approach/one_dimension/line/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line_b $MESH_PATH $SOLUTION_PATH
    
the  fields in this problem are 
    - v_bar == v^{bar}_{Lagrangian approach}
    - w_bar == w^{bar}_{Lagrangian approach}
    - phi == \phi_{Lagrangian approach}
    - v_n == v^{n}_{Lagrangian approach}
    - w_n == w^{n}_{Lagrangian approach}
    - u_n_12[alpha] == X^{n-1/2, alpha}_{Lagrangian approach} - X_ref^alpha
    - X_ref^alpha is the manifold in the reference configuration 
    - nu_n_12 == nu^{n-1/2}_{Lagrangian approach}
    - psi_n_12 == psi^{n-1/2}_{Lagrangian approach}
    - mu_n_12 == mu^{n-1/2}_{Lagrangian approach}
'''


from fenics import *
import importlib
import dolfin

import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import function_spaces as fsp
import parameters.read.solution as rpam
import switch_problem as swi
import variational_problem.utils as var_pr


prout_bc = importlib.import_module(swi.pr_bc)
prout_sol = importlib.import_module(swi.pr_sol)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


set_log_level(20)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

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
PETScOptions.set('snes_atol', 1e-12)  
PETScOptions.set('snes_rtol', 1e-12) 
PETScOptions.set('snes_stol', 1e-8)   
PETScOptions.set('snes_max_it', 100000)
PETScOptions.set('snes_monitor')
PETScOptions.set('snes_max_funcs', 1000000)       


fsp.sigma_n_32.interpolate( vp.sigma_n_32_0_Expression( element=fsp.Q_psi_n_12.ufl_element() ))

#Option 1: set initial profiles

fsp.v_bar_0.interpolate( vp.v_n_0_Expression( element=fsp.Q_v_bar.ufl_element() ) )
fsp.v_n_0.interpolate( vp.v_n_0_Expression( element=fsp.Q_v_n.ufl_element() ) )
fsp.nu_n_12_0.interpolate( vp.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ) )
fsp.u_n_12_0.interpolate( vp.u_n_12_0_Expression( element=fsp.Q_u_n_12.ufl_element() ) )


#Option 2:read initial profiles by reading them from file
# uncomment this to set the initial profiles from the ODE soltion
'''
print("Reading the initial profiles from file ...")
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'v.csv', fsp.v_bar_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'w.csv', fsp.w_bar_0)
# fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'phi.csv', fsp.phi_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'v.csv', fsp.v_n_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'w.csv', fsp.w_n_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'u.csv', fsp.u_n_12_0)
# fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'nu.csv', fsp.nu_n_12_0)
fsp.nu_n_12_0.interpolate( vp.nu_n_12_0_Expression( element=fsp.Q_nu_n_12.ufl_element() ))
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'psi.csv', fsp.psi_n_12_0)
fu.read_from_file(io.add_trailing_slash(rpam.parameters['solution_ode_path']) + 'mu.csv', fsp.mu_n_12_0)
print('... done')
'''

fsp.assigner.assign(fsp.psi, [fsp.v_bar_0, fsp.w_bar_0, fsp.phi_0, fsp.v_n_0, fsp.w_n_0, fsp.u_n_12_0, fsp.nu_n_12_0, fsp.psi_n_12_0, fsp.mu_n_12_0 ])



# Time-stepping
t = 0
for step in range(rpam.parameters['N']):

    print("\n* step = ", step, "\n",flush=True)

    # Update current time
    t += vp.dt

    vp = importlib.import_module(swi.vp)

    # solve the variational problem
    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi, parameters=params)

    #update previous solution:
    #get the solution and write it to file
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, u_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi.split( deepcopy=True )

    
    prout_bc.print_bcs( fsp.psi )
    
    if step % rpam.parameters['print_out_stride'] == 0:
        # step is a multiple of rpam.parameters['print_out_stride'] -> print the solution. This is done in order not to produce too many files in the output
        prout_sol.print_solution(fsp.psi, step, t)
    

    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign( v_n_output )

    fsp.w_n_1.assign( w_n_output )

    fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
    fsp.sigma_n_32.assign(fsp.sigma_n_12)

    fsp.u_n_32.assign( u_n_12_output )


prout_bc.csvfile_bcs.close()
