import dolfin
from fenics import *

import mesh.load as lmsh

# Define function spaces
# the '2' in ''P', 2)' is the order of the polynomials used to describe these spaces: if they are low, then derivatives high enough of the functions projected on thee spaces will be set to zero !
Q_v = VectorFunctionSpace(lmsh.mesh, 'P', 2, dim=2)
Q = FunctionSpace(lmsh.mesh, 'P', 1)

# Define functions for solutions at previous and current time steps
v_n_1 = Function(Q_v)
v_n_2 = Function(Q_v)
v_n = Function(Q_v)
v_ = Function(Q_v)
# sigma^{n-1/2}
sigma_n_12 = Function(Q)
# sigma^{n-3/2}
sigma_n_32 = Function(Q)
phi = Function(Q)

# Define test functions
nu = TestFunction(Q_v)
J_v_ = TrialFunction(Q_v)
J_v_n = TrialFunction(Q_v)
J_phi = TrialFunction(Q)
q = TestFunction(Q)

V = 0.5 * (v_n_1 + v_)
