'''
this code reads a sequence of .h5 files, collates them into a time series in xdmf format and writes it into an xdmf file
run with clear; clear; python3 example.py /home/fenics/shared/mesh/membrane_mesh/ /home/fenics/shared/navier_stokes/ [number of .h5 files to be read]
'''

from __future__ import print_function
from fenics import *
import numpy as np
import time
from geometry import *

XDMF_file_v = XDMFFile( 'solution/v.xdmf' )
XDMF_file_w = XDMFFile( 'solution/w.xdmf' )
XDMF_file_sigma = XDMFFile( 'solution/sigma.xdmf' )
XDMF_file_omega = XDMFFile( 'solution/omega.xdmf' )
XDMF_file_z = XDMFFile( 'solution/z.xdmf' )

v = Function(Q_v_n)
w = Function(Q_w_n)
sigma = Function(Q_phi)
omega = Function(Q_omega_n)
z = Function(Q_z_n)

# Time-stepping
for step in range(1, N+1):
    # time.sleep( 1 )  # Makes Python wait for 5 seconds

    print("\n* step = ", step, "\n")

    # Read the contents of the .h5 files and write them in v, w, .... :
    HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n" + str(step) + ".h5", "r" ).read(v, "/f" )
    HDF5File( MPI.comm_world, "solution/snapshots/h5/w_n" + str(step) + ".h5", "r" ).read(w, "/f" )
    HDF5File( MPI.comm_world, "solution/snapshots/h5/sigma_n" + str(step) + ".h5", "r" ).read(sigma, "/f" )
    HDF5File( MPI.comm_world, "solution/snapshots/h5/omega_n" + str(step) + ".h5", "r" ).read(omega, "/f" )
    HDF5File( MPI.comm_world, "solution/snapshots/h5/z_n" + str(step) + ".h5", "r" ).read(z, "/f" )

    # append into the xdmf files the current time step stored in v, w, ...
    XDMF_file_v.write(v, step)
    XDMF_file_w.write(w, step)
    XDMF_file_sigma.write(sigma, step)
    XDMF_file_omega.write(omega, step)
    XDMF_file_z.write(z, step)

    # HDF5_file_write = HDF5File( MPI.comm_world, "solution/snapshots/h5/v_n" + str(step) + ".h5", "w" )
    # HDF5_file_write.write( v, "/f" )
    # HDF5_file_write.close()