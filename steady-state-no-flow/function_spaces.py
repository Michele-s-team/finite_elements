from fenics import *
from mshr import *
#import read_mesh_square as rmsh
# import read_mesh_ring as rmsh
import read_mesh_square_no_circle as rmsh

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


element = MixedElement( [P_z, P_omega, P_mu, P_nu] )
#total function space
Q = FunctionSpace(rmsh.mesh, element)
#function spaces for z, omega, eta and theta
Q_z= Q.sub( 0 ).collapse()
Q_omega = Q.sub( 1 ).collapse()
Q_mu = Q.sub( 2 ).collapse()
Q_nu = Q.sub( 3 ).collapse()

Q_sigma = FunctionSpace( rmsh.mesh, 'P', 1 )
#the function space for tau is for post-processing only
Q_tau = FunctionSpace( rmsh.mesh, 'P', degree_function_space )

# Define functions
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_z, nu_omega, nu_mu, nu_nu = TestFunctions( Q )

J_pp = TrialFunction( Q_tau )
nu_tau = TestFunction(Q_tau)

#these functions are used to print the solution to file
sigma = Function(Q_sigma)
tau = Function(Q_tau)

z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function( Q_mu )
nu_output = Function( Q_nu )

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

z, omega, mu, nu = split( psi )
assigner = FunctionAssigner( Q, [Q_z, Q_omega, Q_mu, Q_nu] )