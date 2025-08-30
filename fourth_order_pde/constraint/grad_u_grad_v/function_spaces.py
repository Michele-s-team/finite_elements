from fenics import *

import mesh.load as lmsh
import read_parameters_solve as rpam

P_z = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
P_u = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
P_omega_z = VectorElement('P', triangle, rpam.parameters['function_space_degree'])
P_omega_u = VectorElement('P', triangle, rpam.parameters['function_space_degree'])
P_mu = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
element = MixedElement([P_z, P_u, P_omega_z, P_omega_u, P_mu])

Q = FunctionSpace(lmsh.mesh, element)

Q_z = Q.sub(0).collapse()
Q_u = Q.sub(1).collapse()
Q_omega_z = Q.sub(2).collapse()
Q_omega_u = Q.sub(3).collapse()
Q_mu = Q.sub(4).collapse()

# Define variational problem
psi = Function(Q)
nu_z, nu_u, nu_omega_z, nu_omega_u, nu_mu = TestFunctions(Q)

z_output = Function(Q_z)
u_output = Function(Q_u)
omega_z_output = Function(Q_omega_z)
omega_u_output = Function(Q_omega_u)
mu_output = Function(Q_mu)

z_exact = Function(Q_z)
u_exact = Function(Q_u)
omega_z_exact = Function(Q_omega_z)
omega_u_exact = Function(Q_omega_u)
mu_exact = Function(Q_mu)

# functions to store the initial condition for the solver
z_0 = Function(Q_z)
u_0 = Function(Q_u)
omega_z_0 = Function(Q_omega_z)
omega_u_0 = Function(Q_omega_u)
mu_0 = Function(Q_mu)


f = Function(Q_z)
g = Function(Q_omega_z)
J_Q = TrialFunction(Q)
z, u, omega_z, omega_u, mu = split(psi)

assigner = FunctionAssigner(Q, [Q_z, Q_u, Q_omega_z, Q_omega_u, Q_mu])
