import dolfin
from fenics import *

import load_mesh as lmsh

'''
the variables for the problem are
- 'theta'
- 'omega': scalar functions of t
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
Q_v_ = VectorFunctionSpace(lmsh.mesh, 'P', 2, dim=2)
Q_phi = FunctionSpace(lmsh.mesh, 'P', 1)
Q_u = VectorFunctionSpace(lmsh.mesh, 'P', 1)
Q_u_dot = VectorFunctionSpace(lmsh.mesh, 'P', 1)


# Define functions for solutions at previous and current time steps
v_n = Function(Q_v)
v_n_1 = Function(Q_v)
v_n_2 = Function(Q_v)
v_ = Function(Q_v_)
# sigma^{n-1/2}
sigma_n_12 = Function(Q_phi)
# sigma^{n-3/2}
sigma_n_32 = Function(Q_phi)
phi = Function(Q_phi)
u = Function(Q_u)
u_dot = Function(Q_u_dot)

# Define test functions
nu_v_n = TestFunction(Q_v)
nu_v_ = TestFunction(Q_v_)
nu_phi = TestFunction(Q_phi)
nu_u = TestFunction(Q_u)
nu_u_dot = TestFunction(Q_u_dot)


# Jacobians
J_v_ = TrialFunction(Q_v_)
J_v_n = TrialFunction(Q_v)
J_phi = TrialFunction(Q_phi)
J_u = TrialFunction(Q_u)
J_u_dot = TrialFunction(Q_u_dot)

V = 0.5 * (v_n_1 + v_)
