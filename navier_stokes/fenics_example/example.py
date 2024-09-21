from __future__ import print_function
from fenics import *
import numpy as np

mesh = UnitSquareMesh(96, 96)
V = FunctionSpace(mesh, "Lagrange", 1)

v  = Function( V )



# Create XDMF files for visualization output
xdmffile_v = XDMFFile( 'v.xdmf' )

# Time-stepping
for step in range(10):

    print("\n* step = ", step, "\n")

    class AnalyticalExpression( UserExpression ):
        def eval(self, values, x):
            values[0] = step * x[0]

        def value_shape(self):
            return (1,)

    v.interpolate( AnalyticalExpression( element=V.ufl_element() ))

    xdmffile_v.write( v, step )
