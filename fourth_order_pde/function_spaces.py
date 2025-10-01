from fenics import *


import mesh.utils as msh
import mesh.load as lmsh


# CHANGE PARAMETERS HERE
function_space_degree = 4
# CHANGE PARAMETERS HERE

element_geometry = msh.element_geometry(lmsh.mesh)

P_z = FiniteElement('P', element_geometry, function_space_degree)
P_omega = VectorElement('P', element_geometry, function_space_degree)
P_mu = FiniteElement('P', element_geometry, function_space_degree)

element = MixedElement([P_z, P_omega, P_mu])


P_rho = VectorElement('P', element_geometry, function_space_degree)
P_tau = FiniteElement('P', element_geometry, function_space_degree)

element_pp = MixedElement([P_rho, P_tau])

Q = FunctionSpace(lmsh.mesh, element)
Q_pp = FunctionSpace(lmsh.mesh, element_pp)

Q_z = Q.sub(0).collapse()
Q_omega = Q.sub(1).collapse()
Q_mu = Q.sub(2).collapse()

Q_rho = Q_pp.sub(0).collapse()
Q_tau = Q_pp.sub(1).collapse()

# Define variational problem
psi = Function(Q)
psi_pp = Function(Q_pp)

nu_z, nu_omega, nu_mu = TestFunctions(Q)
nu_rho, nu_tau = TestFunctions(Q_pp)

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


z_0 = Function(Q_z)
omega_0 = Function(Q_omega)
mu_0 = Function(Q_mu)

rho_0 = Function(Q_rho)
tau_0 = Function(Q_tau)


f = Function(Q_z)
J_Q = TrialFunction(Q)
J_Q_pp = TrialFunction(Q_pp)

z, omega, mu = split(psi)
rho, tau = split(psi_pp)
