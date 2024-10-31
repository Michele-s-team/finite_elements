'''
this code reads a sequence of .h5 files, collates them into a time series in xdmf format and writes it into an xdmf file
run with
clear; clear; python3 example.py [path of mesh] [path of old solution] [path of new solution]  [number of .h5 files to be read]
clear; clear; rm -rf solution-new; python3 example.py /home/fenics/shared/mesh/membrane_mesh /home/fenics/shared/navier_stokes/fenics_example/solution/snapshots/h5  /home/fenics/shared/navier_stokes/fenics_example/solution-new 5855
'''

from __future__ import print_function
from fenics import *
import numpy as np
import time
from geometry import *

print("mesh old folder =", args.mesh_old_directory)
print("solution old folder =", args.solution_old_directory)
print("solution new folder =", args.solution_new_directory)

# XDMF_file_v = XDMFFile((args.solution_new_directory) +  '/v.xdmf' )
# XDMF_file_w = XDMFFile((args.solution_new_directory) +  '/w.xdmf' )
# XDMF_file_sigma = XDMFFile((args.solution_new_directory) +  '/sigma.xdmf' )
# XDMF_file_omega = XDMFFile((args.solution_new_directory) +  '/omega.xdmf' )
XDMF_file_z = XDMFFile((args.solution_new_directory) +  '/z.xdmf' )

v = Function(Q_v_n)
w = Function(Q_w_n)
sigma = Function(Q_phi)
omega = Function(Q_omega_n)
z = Function(Q_z_n)

# Time-stepping
for step in range(1, N):
    # time.sleep( 1 )  # Makes Python wait for 5 seconds

    print("* step = ", step, "\n")

    # Read the contents of the .h5 files and write them in v, w, .... :
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/v_n_" + str(step) + ".h5", "r" ).read(v, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/w_n_" + str(step) + ".h5", "r" ).read(w, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/sigma_n_12_" + str(step) + ".h5", "r" ).read(sigma, "/f" )
    # HDF5File( MPI.comm_world, (args.solution_old_directory) + "/omega_n_12_" + str(step) + ".h5", "r" ).read(omega, "/f" )
    HDF5File( MPI.comm_world, (args.solution_old_directory) + "/z_n_12_" + str(step) + ".h5", "r" ).read(z, "/f" )

    # append into the xdmf files the current time step stored in v, w, ...
    # XDMF_file_v.write(v, step)
    # XDMF_file_w.write(w, step)
    # XDMF_file_sigma.write(sigma, step)
    # XDMF_file_omega.write(omega, step)
    XDMF_file_z.write(z, step)

    # HDF5_file_write = HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n" + str(step) + ".h5", "w" )
    # HDF5_file_write.write( v, "/f" )
    # HDF5_file_write.close()