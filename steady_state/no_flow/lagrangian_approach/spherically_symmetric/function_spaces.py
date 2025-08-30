from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam

# Define function spaces
'''
the fields are
- psi = psi_notes
- omega = \partial_1 psi
- rho = rho_notes
- zeta = \zeta_notes
'''

P_psi = FiniteElement('P', triangle, rpam.parameters['degree_function_space'])
P_omega = FiniteElement('P', triangle, rpam.parameters['degree_function_space'])
P_rho = FiniteElement('P', triangle, rpam.parameters['degree_function_space'])
P_zeta = FiniteElement('P', triangle, rpam.parameters['degree_function_space'])

element = MixedElement([P_psi, P_omega, P_rho, P_zeta])
# total function space
Q = FunctionSpace(lmsh.mesh, element)
Q_psi = Q.sub(0).collapse()
Q_omega = Q.sub(1).collapse()
Q_rho = Q.sub(2).collapse()
Q_zeta = Q.sub(3).collapse()

Q_sigma = FunctionSpace(lmsh.mesh, 'P', 1)

# Define functions
J_psi = TrialFunction(Q)
phi = Function(Q)
nu_psi, nu_omega, nu_rho, nu_zeta = TestFunctions(Q)

# these functions are used to print the solution to file
sigma = Function(Q_sigma)

psi_exact = Function(Q_psi)
omega_exact = Function(Q_omega)
rho_exact = Function(Q_rho)
zeta_exact = Function(Q_zeta)

# omega_0, z_0 are used to store the initial conditions
psi_0 = Function(Q_psi)
omega_0 = Function(Q_omega)
rho_0 = Function(Q_rho)
zeta_0 = Function(Q_zeta)

psi_0_read = Function(Q_psi)
omega_0_read = Function(Q_omega)
rho_0_read = Function(Q_rho)
zeta_0_read = Function(Q_zeta)

psi, omega, rho, zeta = split(phi)
assigner = FunctionAssigner(Q, [Q_psi, Q_omega, Q_rho, Q_zeta])
