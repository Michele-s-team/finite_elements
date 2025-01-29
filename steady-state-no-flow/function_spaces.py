from fenics import *
from mshr import *
# from read_mesh_square import *
import read_mesh_ring as rmsh
# from read_mesh_square_no_circle import *

# Define function spaces
#finite elements for sigma .... omega
'''
z
omega_i = \partial_i z
mu = H(omega)
nu_i = Nabla_i mu (Nabla is the covariant derivative)
tau = Nabla_i nu^i 
'''
degree_function_space = 2
P_z = FiniteElement( 'P', triangle, degree_function_space )
P_omega = VectorElement( 'P', triangle, degree_function_space )
P_mu = FiniteElement( 'P', triangle, degree_function_space )
P_nu = VectorElement( 'P', triangle, degree_function_space )
P_tau = FiniteElement( 'P', triangle, degree_function_space )

element = MixedElement( [P_z, P_omega, P_mu, P_nu, P_tau] )
#total function space
Q = FunctionSpace(rmsh.mesh, element)
#function spaces for z, omega, eta and theta
Q_z= Q.sub( 0 ).collapse()
Q_omega = Q.sub( 1 ).collapse()
Q_mu = Q.sub( 2 ).collapse()
Q_nu = Q.sub( 3 ).collapse()
Q_tau = Q.sub( 4 ).collapse()

Q_sigma = FunctionSpace( rmsh.mesh, 'P', 1 )

# Define functions
# the Jacobian
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_z, nu_omega, nu_mu, nu_nu, nu_tau = TestFunctions( Q )

#these functions are used to print the solution to file
sigma = Function(Q_sigma)

z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function( Q_mu )
nu_output = Function( Q_nu )
tau_output = Function( Q_tau )

z_exact = Function( Q_z )
omega_exact = Function( Q_omega )
mu_exact = Function( Q_mu )
nu_exact = Function( Q_nu )
tau_exact = Function( Q_tau )

# omega_0, z_0 are used to store the initial conditions
z_0 = Function( Q_z )
omega_0 = Function( Q_omega )
mu_0 = Function( Q_mu )
nu_0 = Function( Q_nu )
tau_0 = Function( Q_tau )

z, omega, mu, nu, tau = split( psi )
assigner = FunctionAssigner( Q, [Q_z, Q_omega, Q_mu, Q_nu, Q_tau] )