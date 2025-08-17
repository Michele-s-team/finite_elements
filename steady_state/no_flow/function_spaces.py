from fenics import *
import dolfin

import load_mesh as lmsh

# Define function spaces
#finite elements for sigma .... omega
'''
z
omega_i = \partial_i z
mu = H(omega)
tau = Nabla_i nu^i 
'''
degree_function_space = 2
P_z = FiniteElement( 'P', triangle, degree_function_space )
P_omega = VectorElement( 'P', triangle, degree_function_space )
P_mu = FiniteElement( 'P', triangle, degree_function_space )


element = MixedElement( [P_z, P_omega, P_mu] )
#total function space
Q = FunctionSpace(lmsh.mesh, element)
#function spaces for z, omega, eta and theta
Q_z= Q.sub( 0 ).collapse()
Q_omega = Q.sub( 1 ).collapse()
Q_mu = Q.sub( 2 ).collapse()

Q_sigma = FunctionSpace( lmsh.mesh, 'P', 1 )
#the function space for tau is for post-processing only
Q_tau = FunctionSpace( lmsh.mesh, 'P', degree_function_space )

# function spaces for the tangential and normal forces per unit length
Q_dFfl_t = VectorFunctionSpace(lmsh.mesh, 'P', degree_function_space)
Q_dFfl_n = FunctionSpace(lmsh.mesh, 'P', degree_function_space)

#function space for three-dimensional vector fields depending on the two coordinates on the mesh
Q_3d = VectorFunctionSpace( lmsh.mesh, 'P', degree_function_space, dim=3 )


'''
function spaces of polynomial order 1 (which should not be changed) which are used to read in functions and assign their nodal values from a list 
as in function.set_from_list 
'''
Q_read = FunctionSpace( lmsh.mesh, 'P', 1 )


# Define functions
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_z, nu_omega, nu_mu = TestFunctions( Q )

J_pp_tau = TrialFunction( Q_tau )
nu_tau = TestFunction(Q_tau)

#these functions are used to print the solution to file
sigma = Function(Q_sigma)
tau = Function(Q_tau)

z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function( Q_mu )

z_exact = Function( Q_z )
omega_exact = Function( Q_omega )
mu_exact = Function( Q_mu )

tau_exact = Function( Q_tau )

#functions used to store the nodal values read from a list or file
z_0_read = Function( Q_read )
omega_0_r_read = Function( Q_read )
mu_0_read = Function( Q_read )


# omega_0, z_0 are used to store the initial conditions
z_0 = Function( Q_z )
omega_0 = Function( Q_omega )
mu_0 = Function( Q_mu )

tau_0 = Function( Q_tau )

z, omega, mu = split( psi )
assigner = FunctionAssigner( Q, [Q_z, Q_omega, Q_mu] )