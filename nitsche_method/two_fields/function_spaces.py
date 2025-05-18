import dolfin
from fenics import *

import load_mesh as lmsh

Q = FunctionSpace(lmsh.mesh, 'P', 2)
V = VectorFunctionSpace(lmsh.mesh, 'P', 2)

# Define variational problem
u = Function(V)
v = TestFunction(V)
f = Function(V)
grad_u_0 = Function(V)
grad_u_1 = Function(V)
g = Function(V)
