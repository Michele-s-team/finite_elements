from fenics import *
from mshr import *
from read_mesh import *


# Define function spaces
#finite elements for sigma .... omega
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_sigma_n = FiniteElement( 'P', triangle, 1 )
P_omega_n = VectorElement( 'P', triangle, 3 )
P_z_n = FiniteElement( 'P', triangle, 1 )

element = MixedElement( [P_v_n, P_w_n, P_sigma_n, P_omega_n, P_z_n] )
#total function space
Q = FunctionSpace(mesh, element)
#function spaces for vbar .... zn
Q_v = Q.sub( 0 ).collapse()
Q_w = Q.sub( 1 ).collapse()
Q_sigma = Q.sub( 2 ).collapse()
Q_omega = Q.sub( 3 ).collapse()
Q_z= Q.sub( 4 ).collapse()

# Define functions
# the Jacobian
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_v, nu_w, nu_sigma, nu_omega, nu_z = TestFunctions( Q )
# fields at the preceeding steps
# v_n_1 = Function(Q_v_n)
# v_n_2 = Function(Q_v_n)
# w_n_1 = Function(Q_w_n)
# sigma_n_12 = Function( Q_phi )
# sigma_n_32 = Function( Q_phi )
# z_n_32 = Function( Q_z_n )

# v_n_0, ...., z_n_0 are used to store the initial conditions
# sigma_n_12_0 = Function( Q_phi )
# v_0 = Function( Q_v )
# w_0 = Function( Q_w )
# sigma_0 = Function( Q_sigma )
# z_0 = Function( Q_z )
# omega_0 = Function( Q_omega )

v, w, sigma, omega, z = split( psi )
