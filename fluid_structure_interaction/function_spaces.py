import dolfin
from fenics import *

import load_mesh as lmsh

'''
the variables for the problem are
- 'theta', 'omega': scalar functions of t
- 'v^n' = \textrm{v}^n_notes
- 'v_' = \textrm{v}^*_notes
- 'phi' = phi_notes
- 'sigma' = \varsigma_notes
- 'u', 'u_dot' = u_notes, \dot{u}_notes
'''

theta = 0
omega = 0

# Define function spaces
Q_v = VectorFunctionSpace(lmsh.mesh, 'P', 2, dim=2)
Q_phi = FunctionSpace(lmsh.mesh, 'P', 1)
Q_u = VectorFunctionSpace(lmsh.mesh, 'P', 1)
Q_u_dot = VectorFunctionSpace(lmsh.mesh, 'P', 1)


# Define functions for solutions at previous and current time steps
v_n_1 = Function(Q_v)
v_n_2 = Function(Q_v)
v_n = Function(Q_v)
# this is v^*_notes
v_ = Function(Q_v)
# sigma^{n-1/2}
sigma_n_12 = Function(Q_phi)
# sigma^{n-3/2}
sigma_n_32 = Function(Q_phi)
phi = Function(Q_phi)

# Define test functions
nu = TestFunction(Q_v)
J_v_ = TrialFunction(Q_v)
J_v_n = TrialFunction(Q_v)
J_phi = TrialFunction(Q_phi)
q = TestFunction(Q_phi)

V = 0.5 * (v_n_1 + v_)
