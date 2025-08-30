import dolfin
from fenics import *

import differential_geometry.boundary.geometry as bgeo
import load_mesh as lmsh


# CHANGE PARAMETERS HERE
function_space_degree = 4
# CHANGE PARAMETERS HERE

P_z = FiniteElement('P', triangle, function_space_degree)
P_omega = VectorElement('P', triangle, function_space_degree)
P_mu = FiniteElement('P', triangle, function_space_degree)
P_rho = VectorElement('P', triangle, function_space_degree)
P_tau = FiniteElement('P', triangle, function_space_degree)
element = MixedElement([P_z, P_omega, P_mu, P_rho, P_tau])
Q = FunctionSpace(lmsh.mesh, element)

Q_z = Q.sub(0).collapse()
Q_omega = Q.sub(1).collapse()
Q_mu = Q.sub(2).collapse()
Q_rho = Q.sub(3).collapse()
Q_tau = Q.sub(4).collapse()

# Define variational problem
psi = Function(Q)
nu_z, nu_omega, nu_mu, nu_rho, nu_tau = TestFunctions(Q)

z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function(Q_mu)
rho_output = Function(Q_rho)
tau_output = Function(Q_tau)

z_exact = Function(Q_z)
omega_exact = Function(Q_omega)
mu_exact = Function(Q_mu)
rho_exact = Function(Q_rho)
tau_exact = Function(Q_tau)

f = Function(Q_z)
J_Q = TrialFunction(Q)
z, omega, mu, rho, tau = split(psi)
