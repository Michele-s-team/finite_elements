'''
this code reads a sequence of .h5 files, collates them into a time series in xdmf format and writes it into an xdmf file
run with
clear; clear; python3 example.py [path of mesh] [path of solution to be read] [path of solution to write]  [number of .h5 files to be read] [increment with which to step from one .h5 file to the next one]
clear; clear; rm -rf solution-out; python3 example.py /home/fenics/shared/mesh /home/fenics/shared/solution-in/snapshots/h5  /home/fenics/shared/solution-out 12700 10
'''

from __future__ import print_function
from fenics import *
from geometry import *

print("mesh old folder =", args.mesh_old_directory)
print("solution old folder =", args.solution_old_directory)
print("solution new folder =", args.solution_new_directory)

XDMF_file_v_n = XDMFFile( (args.solution_new_directory) + '/v_n.xdmf' )
XDMF_file_v_bar = XDMFFile( (args.solution_new_directory) + '/v_bar.xdmf' )
XDMF_file_w_n = XDMFFile( (args.solution_new_directory) + '/w_n.xdmf' )
XDMF_file_w_bar = XDMFFile( (args.solution_new_directory) + '/w_bar.xdmf' )
# XDMF_file_phi = XDMFFile( (args.solution_new_directory) + '/phi.xdmf' )
XDMF_file_sigma_n_12 = XDMFFile( (args.solution_new_directory) + '/sigma_n_12.xdmf' )
XDMF_file_omega_n_12 = XDMFFile( (args.solution_new_directory) + '/omega_n_12.xdmf' )
XDMF_file_z_n_12 = XDMFFile( (args.solution_new_directory) + '/z_n_12.xdmf' )

v_n = Function( Q_v_n )
v_bar = Function(Q_v_bar)
w_n = Function( Q_w_n )
w_bar = Function(Q_w_bar)
sigma_n_12 = Function( Q_phi )
# phi = Function( Q_phi )
omega_n_12 = Function( Q_omega_n )
z_n_12 = Function( Q_z_n )

# Time-stepping
for step in range(1, N, increment):
    # time.sleep( 1 )  # Makes Python wait for 5 seconds

    print("* step = ", step, "\n")

    # Read the contents of the .h5 files and write them in v, w, .... :
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/v_n_" + str(step) + ".h5", "r" ).read( v_n, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/v_bar_" + str(step) + ".h5", "r" ).read(v_bar, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/w_n_" + str(step) + ".h5", "r" ).read( w_n, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/w_bar_" + str(step) + ".h5", "r" ).read(w_bar, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/phi_" + str(step) + ".h5", "r" ).read( phi, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/sigma_n_12_" + str(step) + ".h5", "r" ).read( sigma_n_12, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/omega_n_12_" + str(step) + ".h5", "r" ).read( omega_n_12, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/z_n_12_" + str(step) + ".h5", "r" ).read( z_n_12, "/f" )

    # append into the xdmf files the current time step stored in v, w, ...
    XDMF_file_v_n.write( v_n, step )
    XDMF_file_v_n.write( v_bar, step )
    XDMF_file_w_n.write( w_n, step )
    XDMF_file_w_n.write( w_bar, step )
    # XDMF_file_phi.write( phi, step )
    XDMF_file_sigma_n_12.write( sigma_n_12, step )
    XDMF_file_omega_n_12.write( omega_n_12, step )
    XDMF_file_z_n_12.write( z_n_12, step )

    # HDF5_file_write = HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n" + str(step) + ".h5", "w" )
    # HDF5_file_write.write( v, "/f" )
    # HDF5_file_write.close()