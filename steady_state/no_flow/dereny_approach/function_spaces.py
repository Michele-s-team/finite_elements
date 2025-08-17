from fenics import *

import load_mesh as lmsh
import read_parameters_solve as rpam

# Define function spaces
# finite elements for psi, rho, zeta

P_psi = FiniteElement('P', triangle, rpam.parameters['degree_function_space'])
P_rho = FiniteElement('P', triangle, rpam.parameters['degree_function_space'])
P_zeta = FiniteElement('P', triangle, rpam.parameters['degree_function_space'])

element = MixedElement([P_psi, P_rho, P_zeta])
# total function space
Q = FunctionSpace(lmsh.mesh, element)
# function spaces for z, omega, eta and theta
Q_psi = Q.sub(0).collapse()
Q_rho = Q.sub(1).collapse()
Q_zeta = Q.sub(2).collapse()

Q_sigma = FunctionSpace(lmsh.mesh, 'P', 1)


# Define functions
J_psi = TrialFunction(Q)
phi = Function(Q)
nu_psi, nu_rho, nu_zeta = TestFunctions(Q)


# these functions are used to print the solution to file
sigma = Function(Q_sigma)

psi_output = Function(Q_psi)
rho_output = Function(Q_rho)
zeta_output = Function(Q_zeta)

psi_exact = Function(Q_psi)
rho_exact = Function(Q_rho)
zeta_exact = Function(Q_zeta)



# omega_0, z_0 are used to store the initial conditions
psi_0 = Function(Q_psi)
rho_0 = Function(Q_rho)
zeta_0 = Function(Q_zeta)


psi, rho, zeta = split(phi)
assigner = FunctionAssigner(Q, [Q_psi, Q_rho, Q_zeta])
