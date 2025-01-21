from symtable import Function
from fenics import *
from mshr import *
# from read_mesh_square import *
from read_mesh_ring import *
# from read_mesh_square_no_circle import *

# Define function spaces
#finite elements for sigma .... omega
'''
z
omega_i = \partial_i z
mu = H(omega)
nu_i = Nabla_i eta (where Nabla is the covariant derivative)
'''
degree_function_space = 2
P_z = FiniteElement( 'P', triangle, degree_function_space )
P_omega = VectorElement( 'P', triangle, degree_function_space )
P_mu = FiniteElement( 'P', triangle, degree_function_space )
P_nu = VectorElement( 'P', triangle, degree_function_space )

element = MixedElement( [P_z, P_omega, P_mu, P_nu] )
#total function space
Q = FunctionSpace(mesh, element)
#function spaces for z, omega, eta and theta
Q_z= Q.sub( 0 ).collapse()
Q_omega = Q.sub( 1 ).collapse()
Q_mu = Q.sub( 2 ).collapse()
Q_nu = Q.sub( 3 ).collapse()

Q_sigma = FunctionSpace( mesh, 'P', 1 )

# Define functions
# the Jacobian
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_z, nu_omega, nu_mu, nu_nu = TestFunctions( Q )

#these functions are used to print the solution to file
sigma = Function(Q_sigma)
z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function( Q_mu )
nu_output = Function( Q_nu )

# omega_0, z_0 are used to store the initial conditions
z_0 = Function( Q_z )
omega_0 = Function( Q_omega )
mu_0 = Function( Q_mu )
nu_0 = Function( Q_nu )

z, omega, mu, nu = split( psi )
assigner = FunctionAssigner( Q, [Q_z, Q_omega, Q_mu, Q_nu] )