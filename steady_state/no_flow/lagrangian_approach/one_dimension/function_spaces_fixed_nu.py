from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam



P_psi = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_mu = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_u = VectorElement('P', interval, rpam.parameters['function_space_degree'], dim=2)

element = MixedElement([P_psi, P_mu, P_u])
# total function space
Q = FunctionSpace(lmsh.mesh, element)
# function spaces for psi, mu, u
Q_psi = Q.sub(0).collapse()
Q_mu = Q.sub(1).collapse()
Q_u = Q.sub(2).collapse()

Q_sigma = FunctionSpace(lmsh.mesh, 'P', 1)
Q_nu = FunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'])
Q_X = VectorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'], dim=2)

'''
function spaces of polynomial order 1 (which should not be changed) which are used to read in functions and assign their nodal values from a list 
as in function.set_from_list 
'''
Q_read = FunctionSpace(lmsh.mesh, 'P', 1)

# Define functions
J_phi = TrialFunction(Q)
phi = Function(Q)
nu_psi, nu_mu, nu_u = TestFunctions(Q)

# these functions are used to print the solution to file
sigma = Function(Q_sigma)
nu =  Function(Q_nu)
X_ref = Function(Q_X)

psi_output = Function(Q_psi)
mu_output = Function(Q_mu)
u_output = Function(Q_u)

psi_exact = Function(Q_psi)
mu_exact = Function(Q_mu)
u_exact = Function(Q_u)



# omega_0, z_0 are used to store the initial conditions
psi_0 = Function(Q_psi)
mu_0 = Function(Q_mu)
u_0 = Function(Q_u)

psi, mu, u = split(phi)
assigner = FunctionAssigner(Q, [Q_psi, Q_mu, Q_u])

