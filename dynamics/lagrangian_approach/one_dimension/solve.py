'''
This file solves for the dynamics of a one-dimensional fluid in the generalized arc-length gauge

This file needs the mesh files, which can be generated, for example, by `finite_elements/mesh/generate_mesh.py` with
python3 generate_mesh.py 0.1
and which are stored into finite_elements/mesh/solution

Run with
clear; clear; rm -rf solution; mkdir solution; python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]

Examples:

    MESH_PATH="/home/fenics/shared/generate_mesh/1d/line/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/solution"; rm -rf $SOLUTION_PATH; python3 solve.py line $MESH_PATH $SOLUTION_PATH
'''


import colorama as col
from fenics import *
import importlib
import dolfin

import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import function_spaces as fsp
import input_output as io
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi


prout_bc = importlib.import_module(swi.prout_bc)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


set_log_level(20)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

print("Input diredtory = ", rarg.args.input_directory)
print("Output diredtory = ", rarg.args.output_directory)
print(f"Radius of mesh cell = {col.Fore.BLUE}{rmsh.r_mesh:.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


#Option 1: set initial profiles
'''
fsp.v_n_1.interpolate(vp.TangentVelocityExpression(element=fsp.Q_v_n.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.w_n_1.interpolate(vp.NormalVelocityExpression(element=fsp.Q_w_n.ufl_element()))
fsp.sigma_n_32.interpolate( vp.SurfaceTensionExpression( element=fsp.Q_phi.ufl_element() ))
fsp.z_n_32.interpolate( vp.ManifoldExpression( element=fsp.Q_z_n.ufl_element() ) )
# omega_n_32.interpolate( vp.OmegaExpression( element=fsp.Q_omega_n.ufl_element() ))
'''

#Option 2:read initial profiles by reading them from file
'''
read_step = 400
print("Reading initial condition from file ... ")
HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n_" + str( read_step-1 ) + ".h5", "r" ).read(fsp.v_n_1, "/f" )
HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n_" + str( read_step-2 ) + ".h5", "r" ).read(fsp.v_n_2, "/f" )
HDF5File( MPI.comm_world, "solution/snapshots/h5/w_n_" + str( read_step-1 ) + ".h5", "r" ).read(fsp.w_n_1, "/f" )
HDF5File( MPI.comm_world, "solution/snapshots/h5/sigma_n_12_" + str( read_step-1 ) + ".h5", "r" ).read(fsp.sigma_n_32, "/f" )
HDF5File( MPI.comm_world, "solution/snapshots/h5/z_n_12_" + str( read_step-1 ) + ".h5", "r" ).read(fsp.z_n_32, "/f" )
HDF5File( MPI.comm_world, "solution/snapshots/h5/omega_n_12_" + str( read_step-1 ) + ".h5", "r" ).read(fsp.omega_n_32, "/f" )
HDF5File( MPI.comm_world, "solution/snapshots/h5/mu_n_12_" + str( read_step-1 ) + ".h5", "r" ).read(fsp.mu_n_32, "/f" )
print("... done.")
'''


'''
# Time-stepping
t = 0
for step in range(rpam.parameters['N']):

    print("\n* step = ", step, "\n")

    # Update current time
    t += vp.dt

    vp = importlib.import_module(swi.vp)

    # solve the variational problem
    J = derivative( vp.F, fsp.psi, fsp.J_psi )
    problem = NonlinearVariationalProblem( vp.F, fsp.psi, vp.bcs, J )
    solver = NonlinearVariationalSolver( problem )

  
    #set the solver parameters here
    # params = {'nonlinear_solver': 'newton',
    #            'newton_solver':
    #             {
    #                 'linear_solver'           : 'mumps',
    #                 # 'line_search' : 'bt',
    #                 'absolute_tolerance'      : 1e-6,
    #                 'relative_tolerance'      : 1e-6,
    #                 'maximum_iterations'      : 1000000,
    #                 # 'sign'                    : 'nonnegative',
    #                 'relaxation_parameter'    : 0.95,
    #                 # 'preconditioner'    : 'ilu',
    #                 'lu_solver' :{
    #                     # 'report' : True,
    #                      'symmetric' : False
    #                 },
    #                 'krylov_solver' :{
    #                     'divergence_limit' : 1e0,
    #                     'absolute_tolerance' : 1e-6,
    #                     'relative_tolerance' : 1e-6,
    #                     'nonzero_initial_guess' : True
    #                 }
    # 
    #              }
    # }
    # solver.parameters.update(params)
    
    solver.solve()


    #update previous solution:
    #get the solution and write it to file
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, X_n_12_output, nu_n_12_output, psi_n_12_output, mu_n_12_output = fsp.psi.split( deepcopy=True )

    prout_bc.print_bcs( fsp.psi )
    prout_bc.print_solution( fsp.psi, step, t )


    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign( v_n_output )

    fsp.w_n_1.assign( w_n_output )

    fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
    fsp.sigma_n_32.assign(fsp.sigma_n_12)

    fsp.X_n_32.assign( X_n_12_output )
'''


prout_bc.csvfile_bcs.close()
prout_bc.csvfile_F.close()