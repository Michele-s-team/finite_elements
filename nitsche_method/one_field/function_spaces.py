from fenics import *

import mesh.load as lmsh

V = FunctionSpace(lmsh.mesh, 'P', 8)

# Define variational problem
u = Function(V)
u_D = Function(V)
v = TestFunction(V)
f = Function(V)