from __future__ import print_function
from fenics import *
import numpy as np
import time

mesh = UnitSquareMesh(96, 96)
V = FunctionSpace(mesh, "Lagrange", 1)

v  = Function( V )
w  = Function( V )

#define the expression to be assigned to v
expr = Expression("p*x[0]", degree = 1, p = 1.0)




# Time-stepping
for step in range(10):

    print("\n* step = ", step, "\n")

    # time.sleep( 5 )  # Makes Python wait for 5 seconds

    expr.p = step;
    # this is the correct way to interpolate an expression and write it into v
    v.interpolate(expr)

    HDF5_file_write = HDF5File( MPI.comm_world, "solution/v_n" + str(step) + ".h5", "w" )
    HDF5_file_write.write( v, "/f" )
    HDF5_file_write.close()

    # Read the contents of the file back into a new function, `f2`:
    HDF5_file_read = HDF5File( MPI.comm_world, "solution/v_n" + str(step) + ".h5", "r" )
    HDF5_file_read.read(w, "/f" )
    HDF5_file_read.close()

    # Create XDMF files  and write to it the solution for each time step
    XDMF_file_v = XDMFFile( 'solution/v_n' + str( step ) + '.xdmf' )
    XDMF_file_v.write(w)
