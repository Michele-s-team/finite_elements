'''
This file solves for the dynamics of a two-dimensional fluid

This file needs the mesh files, which can be generated, for example, by `finite_elements/mesh/generate_mesh.py` with
python3 generate_mesh.py 0.1
and which are stored into finite_elements/mesh/solution

Run with
clear; clear; rm -rf solution; mkdir solution; python3 solve.py [name of variational problem] [path where to read the mesh] [path where to store the solution]

Examples:
    rm -rf solution; python3 solve.py square_a /home/fenics/shared/dynamics/mesh/solution /home/fenics/shared/dynamics/solution
    rm -rf solution; mpirun -np 6 python3 solve.py square_a /home/fenics/shared/dynamics/mesh/solution /home/fenics/shared/dynamics/solution
    time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py square_a $MESH $SOLUTION

    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_a $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_b $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/square/symmetric_left_right_top_bottom/solution"; SOLUTION_PATH="/home/fenics/shared/dynamics/solution"; rm -rf $SOLUTION_PATH; python3 solve.py square_b $MESH_PATH $SOLUTION_PATH


To produce figure_10, 
- generate the mesh with /home/fenics/shared/generate_mesh/2d/square/figure_10_parameters/mesh_parameters.csv 
- use parameter file parameters_bc_square_a_figure_10.csv
- run with problem 'square_a'. 
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
import parameters.read.solution as rpam
import switch_problem as swi
import variational_problem.utils as var_pr

prout_bc = importlib.import_module(swi.prout_bc)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


set_log_level(20)
dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

params = {'nonlinear_solver': 'newton',
        'newton_solver':
            {
            'linear_solver': 'default',
            'absolute_tolerance': 1e-10,
            'relative_tolerance': 1e-9,
            'maximum_iterations': 50,
            'relaxation_parameter': None,
            'preconditioner': 'default'
            }
        }


#Option 1: set initial profiles
#
fsp.v_n_1.interpolate(vp.TangentVelocityExpression(element=fsp.Q_v_n.ufl_element()))
fsp.v_n_2.assign(fsp.v_n_1)
fsp.w_n_1.interpolate(vp.NormalVelocityExpression(element=fsp.Q_w_n.ufl_element()))
fsp.sigma_n_32.interpolate( vp.SurfaceTensionExpression( element=fsp.Q_phi.ufl_element() ))
fsp.z_n_32.interpolate( vp.ManifoldExpression( element=fsp.Q_z_n.ufl_element() ) )
# omega_n_32.interpolate( vp.OmegaExpression( element=fsp.Q_omega_n.ufl_element() ))
#

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



# Time-stepping
t = 0
for step in range(rpam.parameters['N']):

    print("\n* step = ", step, "\n")

    # Update current time
    t += vp.dt

    vp = importlib.import_module(swi.vp)


    # solve variational problem
    var_pr.solve_vp(vp.F, fsp.psi, vp.bcs, fsp.J_psi, parameters=params)

    # solve variational problems for post-processing
    var_pr.solve_vp(vp.vp_pp.F_pp_tau, fsp.tau_n_12, vp.vp_pp.bcs_tau, fsp.J_pp_tau, parameters=params)
    var_pr.solve_vp(vp.vp_pp.F_pp_d, fsp.d, vp.vp_pp.bcs_d, fsp.J_pp_d, parameters=params)

    #update previous solution:
    #get the solution and write it to file
    v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, z_n_12_output, omega_n_12_output, mu_n_12_output = fsp.psi.split( deepcopy=True )

    prout_bc.print_bcs( fsp.psi )
    prout_bc.print_solution( fsp.psi, step, t )


    fsp.v_n_2.assign(fsp.v_n_1)
    fsp.v_n_1.assign( v_n_output )

    fsp.w_n_1.assign( w_n_output )

    fsp.sigma_n_12.assign( fsp.sigma_n_32 - project( phi_output, fsp.Q_phi ) )
    fsp.sigma_n_32.assign(fsp.sigma_n_12)

    fsp.z_n_32.assign( z_n_12_output )



prout_bc.csvfile_bcs.close()
prout_bc.csvfile_F.close()