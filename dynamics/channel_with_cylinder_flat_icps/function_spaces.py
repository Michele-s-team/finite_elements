import dolfin
from fenics import *

import load_mesh as lmsh
import boundary_geometry as bgeo

# Define function spaces
V = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q = FunctionSpace(lmsh.mesh, 'P', 1)


# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

U = 0.5 * (u_n + u)
