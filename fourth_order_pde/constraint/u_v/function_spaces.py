from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam

P_z = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
P_u = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
P_omega = VectorElement('P', triangle, rpam.parameters['function_space_degree'])
P_mu = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
element = MixedElement([P_z, P_u, P_omega, P_mu])

Q = FunctionSpace(lmsh.mesh, element)

Q_z = Q.sub(0).collapse()
Q_u = Q.sub(1).collapse()
Q_omega = Q.sub(2).collapse()
Q_mu = Q.sub(3).collapse()

# Define variational problem
psi = Function(Q)
nu_z, nu_u, nu_omega, nu_mu = TestFunctions(Q)

z_output = Function(Q_z)
u_output = Function(Q_u)
omega_output = Function(Q_omega)
mu_output = Function(Q_mu)

z_exact = Function(Q_z)
u_exact = Function(Q_u)
omega_exact = Function(Q_omega)
mu_exact = Function(Q_mu)

# functions to store the initial condition for the solver
z_0 = Function(Q_z)
u_0 = Function(Q_u)
omega_0 = Function(Q_omega)
mu_0 = Function(Q_mu)


f = Function(Q_z)
g = Function(Q_u)
J_Q = TrialFunction(Q)
z, u, omega, mu = split(psi)

assigner = FunctionAssigner(Q, [Q_z, Q_u, Q_omega, Q_mu])
