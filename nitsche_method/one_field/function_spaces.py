import dolfin
from fenics import *

import boundary_geometry as bgeo
import load_2d_mesh as lmsh

V = FunctionSpace(mesh, 'P', 8)

# Define variational problem
u = Function(V)
u_D = Function(V)
v = TestFunction(V)
f = Function(V)