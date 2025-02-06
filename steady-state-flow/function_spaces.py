from fenics import *
from mshr import *

import boundary_geometry as bgeo

import read_mesh_ring as rmsh


# Define function spaces
#finite elements for sigma .... omega
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_sigma_n = FiniteElement( 'P', triangle, 1 )
P_omega_n = VectorElement( 'P', triangle, 3 )
P_z_n = FiniteElement( 'P', triangle, 1 )

element = MixedElement( [P_v_n, P_w_n, P_sigma_n, P_omega_n, P_z_n] )
#total function space
Q = FunctionSpace(bgeo.mesh, element)
#function spaces for vbar .... zn
Q_v = Q.sub( 0 ).collapse()
Q_w = Q.sub( 1 ).collapse()
Q_sigma = Q.sub( 2 ).collapse()
Q_omega = Q.sub( 3 ).collapse()
Q_z= Q.sub( 4 ).collapse()
#function space to store force fields
Q_f = VectorFunctionSpace( bgeo.mesh, 'P', 2 )

# Define functions
# the Jacobian
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_v, nu_w, nu_sigma, nu_omega, nu_z = TestFunctions( Q )

#these functions are used to print the solution to file
v_output = Function(Q_v)
w_output = Function(Q_w)
sigma_output = Function(Q_sigma)
omega_output = Function(Q_omega)
z_output = Function(Q_z)

# v_0, .... are used to store the initial conditions
v_0 = Function( Q_v )
w_0 = Function( Q_w )
sigma_0 = Function( Q_sigma )
omega_0 = Function( Q_omega )
z_0 = Function( Q_z )

v, w, sigma, omega, z = split( psi )
assigner = FunctionAssigner(Q, [Q_v, Q_w, Q_sigma, Q_omega, Q_z])