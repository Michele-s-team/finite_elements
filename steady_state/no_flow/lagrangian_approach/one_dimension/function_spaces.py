from fenics import *

import load_mesh as lmsh
import read_parameters_solve as rpam

'''
the fields in this problem are
psi = psi_{Lagrangian approach}
mu = H
X[i] = {X^i}_{Lagrangian approach}
'''

P_psi = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_mu = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_X = VectorElement('P', interval, rpam.parameters['function_space_degree'], dim=2)

element = MixedElement([P_psi, P_mu, P_X])
# total function space
Q = FunctionSpace(lmsh.mesh, element)
# function spaces for z, omega, eta and theta
Q_psi = Q.sub(0).collapse()
Q_mu = Q.sub(1).collapse()
Q_X = Q.sub(2).collapse()

Q_sigma = FunctionSpace(lmsh.mesh, 'P', 1)

'''
function spaces of polynomial order 1 (which should not be changed) which are used to read in functions and assign their nodal values from a list 
as in function.set_from_list 
'''
Q_read = FunctionSpace(lmsh.mesh, 'P', 1)

# Define functions
J_phi = TrialFunction(Q)
phi = Function(Q)
nu_psi, nu_mu, nu_X = TestFunctions(Q)

# these functions are used to print the solution to file
sigma = Function(Q_sigma)

psi_output = Function(Q_psi)
mu_output = Function(Q_mu)
X_output = Function(Q_X)

psi_exact = Function(Q_psi)
mu_exact = Function(Q_mu)
X_exact = Function(Q_X)

'''
# functions used to store the nodal values read from a list or file
psi_0_read = Function(Q_read)
mu_0_read = Function(Q_read)
X_0_r_read = Function(Q_read)
'''

# omega_0, z_0 are used to store the initial conditions
psi_0 = Function(Q_psi)
mu_0 = Function(Q_mu)
X_0 = Function(Q_X)

psi, mu, X = split(phi)
assigner = FunctionAssigner(Q, [Q_psi, Q_mu, Q_X])
