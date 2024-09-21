from __future__ import print_function
from fenics import *
import numpy as np

mesh = UnitSquareMesh(96, 96)
V = FunctionSpace(mesh, "Lagrange", 1)

v  = Function( V )

#define the expression to be assigned to v
expr = Expression("p*x[0]", degree = 1, p = 1.0)


# Create XDMF files for visualization output
xdmffile_v = XDMFFile( 'v.xdmf' )

# Time-stepping
for step in range(10):

    print("\n* step = ", step, "\n")

    expr.p = step;
    #this is the correct way to interpolate an expression and write it into v
    v.interpolate(expr)

    xdmffile_v.write( v, step )
