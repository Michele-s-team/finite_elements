from fenics import *
import importlib

import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


#define elements and function spaces
P_u = FiniteElement( 'P', triangle, rpam.parameters['function_space_degree'] )
P_v = VectorElement( 'P', triangle, rpam.parameters['function_space_degree'] )
element = MixedElement( [P_u, P_v] )
Q = FunctionSpace( rmsh.mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_v = Q.sub( 1 ).collapse()



# Define functions
psi = Function( Q )
nu_u, nu_v = TestFunctions( Q )

u_output = Function( Q_u )
v_output = Function( Q_v )
u_exact = Function( Q_u )
v_exact = Function( Q_v )
laplacian_u_exact = Function( Q_u )

f = Function( Q_u )
J_uv = TrialFunction( Q )
u, v = split( psi )