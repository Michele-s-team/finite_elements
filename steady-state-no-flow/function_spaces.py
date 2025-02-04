from fenics import *
from mshr import *

import boundary_geometry as bgeo

# import read_mesh_square as rmsh
import read_mesh_ring as rmsh
# import read_mesh_square_no_circle as rmsh


# Define function spaces
#finite elements for sigma .... omega
'''
z
omega_i = \partial_i z
mu = H(omega)
tau = Nabla_i nu^i 
'''
degree_function_space = 1
P_z = FiniteElement( 'P', triangle, degree_function_space )
P_omega = VectorElement( 'P', triangle, degree_function_space )
P_mu = FiniteElement( 'P', triangle, degree_function_space )


element = MixedElement( [P_z, P_omega, P_mu] )
#total function space
Q = FunctionSpace(bgeo.mesh, element)
#function spaces for z, omega, eta and theta
Q_z= Q.sub( 0 ).collapse()
Q_omega = Q.sub( 1 ).collapse()
Q_mu = Q.sub( 2 ).collapse()

Q_sigma = FunctionSpace( bgeo.mesh, 'P', 1 )
#the function spaces for nu and tau are for post-processing only
Q_nu = VectorFunctionSpace( bgeo.mesh, 'P', degree_function_space )
Q_tau = FunctionSpace( bgeo.mesh, 'P', degree_function_space )

# Define functions
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_z, nu_omega, nu_mu = TestFunctions( Q )

J_pp_nu = TrialFunction( Q_nu )
J_pp_tau = TrialFunction( Q_tau )
nu_nu = TestFunction(Q_nu)
nu_tau = TestFunction(Q_tau)

#these functions are used to print the solution to file
sigma = Function(Q_sigma)
nu = Function(Q_nu)
tau = Function(Q_tau)

z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function( Q_mu )

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

z, omega, mu = split( psi )
assigner = FunctionAssigner( Q, [Q_z, Q_omega, Q_mu] )