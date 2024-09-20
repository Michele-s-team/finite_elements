from __future__ import print_function
from fenics import *
import numpy as np

mesh = UnitSquareMesh(96, 96)
V = FunctionSpace(mesh, "Lagrange", 1)

v  = Function( V )

class AnalyticalExpression( UserExpression ):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = x[0]
    def value_shape(self):
        return (1,)

v = interpolate( AnalyticalExpression( element=V.ufl_element() ), V )



# Create XDMF files for visualization output
xdmffile_v_n = XDMFFile( 'v.xdmf' )

# Time-stepping
for step in range(10):

    print("\n* step = ", step, "\n")

    if step>0:
        xdmffile_v_n.write( v, step )
