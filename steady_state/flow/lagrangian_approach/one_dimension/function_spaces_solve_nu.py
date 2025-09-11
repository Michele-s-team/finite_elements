from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam

'''
the fields in this problem are
v[i] = v^i_{Lagrangian approach}
w = w_{Lagrangian approach}
sigma = \sigma_{Lagrangian approach}
psi = psi_{Lagrangian approach}
mu = H
X[i] = {X^i}_{Lagrangian approach}
'''

P_v = VectorElement('P', interval, 2)
P_w = FiniteElement('P', interval, 1)
P_sigma = FiniteElement('P', interval, 1)
P_psi = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_mu = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_X = VectorElement('P', interval, rpam.parameters['function_space_degree'], dim=2)
P_nu = FiniteElement('P', interval, rpam.parameters['function_space_degree'])

element = MixedElement([P_v, P_w, P_sigma, P_psi, P_mu, P_X, P_nu])
# total function space
Q = FunctionSpace(lmsh.mesh, element)
# function spaces for z, omega, eta and theta
Q_v = Q.sub(0).collapse()
Q_w = Q.sub(1).collapse()
Q_sigma = Q.sub(2).collapse()
Q_psi = Q.sub(3).collapse()
Q_mu = Q.sub(4).collapse()
Q_X = Q.sub(5).collapse()
Q_nu = Q.sub(6).collapse()

# function space for the function nu of the arc-length gauge


'''
function spaces of polynomial order 1 (which should not be changed) which are used to read in functions and assign their nodal values from a list 
as in function.set_from_list 
'''
Q_read = FunctionSpace(lmsh.mesh, 'P', 1)

# Define functions
J_phi = TrialFunction(Q)
phi = Function(Q)
nu_v, nu_w, nu_sigma, nu_psi, nu_mu, nu_X, nu_nu = TestFunctions(Q)

v_output = Function(Q_v)
w_output = Function(Q_w)
sigma_output = Function(Q_sigma)
psi_output = Function(Q_psi)
mu_output = Function(Q_mu)
X_output = Function(Q_X)
nu_output = Function(Q_nu)

# omega_0, z_0 are used to store the initial conditions
v_0 = Function(Q_v)
w_0 = Function(Q_w)
sigma_0 = Function(Q_sigma)
psi_0 = Function(Q_psi)
mu_0 = Function(Q_mu)
X_0 = Function(Q_X)
nu_0 = Function(Q_nu)

# functions used to store the nodal values read from a list or file
v_0_read = Function(Q_v)
w_0_read = Function(Q_w)
sigma_0_read = Function(Q_sigma)
psi_0_read = Function(Q_psi)
mu_0_read = Function(Q_mu)
X_0_read = Function(Q_X)
nu_0_read = Function(Q_nu)

v, w, sigma, psi, mu, X, nu = split(phi)
assigner = FunctionAssigner(Q, [Q_v, Q_w, Q_sigma, Q_psi, Q_mu, Q_X, Q_nu])
