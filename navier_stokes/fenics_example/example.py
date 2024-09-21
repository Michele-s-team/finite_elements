from __future__ import print_function
from fenics import *
import numpy as np
import time

mesh = UnitSquareMesh(96, 96)
V = FunctionSpace(mesh, "Lagrange", 1)

v  = Function( V )

#define the expression to be assigned to v
expr = Expression("p*x[0]", degree = 1, p = 1.0)




# Time-stepping
for step in range(10):

    print("\n* step = ", step, "\n")

    time.sleep( 5 )  # Makes Python wait for 5 seconds

    expr.p = step;
    #this is the correct way to interpolate an expression and write it into v
    v.interpolate(expr)

    # Create XDMF files  and write to it the solution for each time step
    xdmffile_v = XDMFFile( 'v_n' + str( step ) + '.xdmf' )
    xdmffile_v.write(v)
