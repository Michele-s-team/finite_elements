import dolfin
from fenics import *

import mesh.load as lmsh

'''
the variables for the problem are
- 'theta_n', 'theta_n_1' : \theta^n, \theta^{n-1} in notes
- 'omega_n', 'omega_n_1' : \omega^n, \omega^{n-1} in notes
- 'v^n' = \textrm{v}^n_notes
- 'v_' = \textrm{v}^*_notes
- 'phi' = phi_notes
- 'sigma' = \varsigma_notes
- 'u', 'u_dot' = u_notes, \dot{u}_notes
'''


theta_n : float
omega_n : float
theta_n_1 : float
omega_n_1 : float

# Define function spaces
Q_v = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q_v_ = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q_phi = FunctionSpace(lmsh.mesh, 'P', 1)

Q_u = VectorFunctionSpace(lmsh.mesh, 'P', 1)
Q_u_dot = VectorFunctionSpace(lmsh.mesh, 'P', 1)

# function space for stress tensor \varsigma
Q_sigma_stress = TensorFunctionSpace(lmsh.mesh, 'P', 2, shape=(2, 2))

# function space for the vector dy(s)/ds which represents the tangent to the ellipse curve
Q_y = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q_dyds = VectorFunctionSpace(lmsh.mesh, 'P', 2)



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
sigma_stress_n = Function(Q_sigma_stress)
u_n = Function(Q_u)
u_dot_n = Function(Q_u_dot)
u_n_1 = Function(Q_u)
u_dot_n_1 = Function(Q_u_dot)
u_n_2 = Function(Q_u)
u_dot_n_2 = Function(Q_u_dot)

u_polygon = Function(Q_u)
u_square = Function(Q_u)
u_dot_polygon = Function(Q_u_dot)
u_dot_square = Function(Q_u_dot)

# y_ellipse = {y^s}_notes
ys_ellipse = Function(Q_y)
# dyds_ellipse = {dy^s/ds}_notes
dyds_ellipse = Function(Q_dyds)

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
