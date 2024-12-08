from symtable import Function
from fenics import *
from mshr import *
# from read_mesh_square import *
from read_mesh_ring import *
# from read_mesh_square_no_circle import *

# Define function spaces
#finite elements for sigma .... omega
P_omega = VectorElement( 'P', triangle, 3 )
P_z = FiniteElement( 'P', triangle, 1 )

element = MixedElement( [P_omega, P_z] )
#total function space
Q = FunctionSpace(mesh, element)
#function spaces for omega and z
Q_omega = Q.sub( 0 ).collapse()
Q_z= Q.sub( 1 ).collapse()

Q_sigma = FunctionSpace( mesh, 'P', 1 )

# Define functions
# the Jacobian
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_omega, nu_z = TestFunctions( Q )

#these functions are used to print the solution to file
sigma = Function(Q_sigma)
omega_output = Function(Q_omega)
z_output = Function(Q_z)

# omega_0, z_0 are used to store the initial conditions
omega_0 = Function( Q_omega )
z_0 = Function( Q_z )

omega, z = split( psi )
assigner = FunctionAssigner(Q, [Q_omega, Q_z])