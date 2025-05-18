import dolfin
from fenics import *

import boundary_geometry as bgeo
import load_mesh as lmsh

# CHANGE PARAMETERS HERE
function_space_degree = 4
# CHANGE PARAMETERS HERE


P_u = FiniteElement( 'P', triangle, function_space_degree )
P_v = FiniteElement( 'P', triangle, function_space_degree )
P_w = FiniteElement( 'P', triangle, function_space_degree )
element = MixedElement( [P_u, P_v, P_w] )
Q = FunctionSpace( lmsh.mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_v = Q.sub( 1 ).collapse()
Q_w = Q.sub( 2 ).collapse()
Q_grad_v = VectorFunctionSpace( lmsh.mesh, 'P', function_space_degree )

psi = Function( Q )
nu_u, nu_v, nu_w = TestFunctions( Q )

grad_v = Function( Q_grad_v )
u_output = Function( Q_u )
v_output = Function( Q_v )
w_output = Function( Q_w )
u_exact = Function( Q_u )
v_exact = Function( Q_v )
w_exact = Function( Q_w )

f = Function( Q_w )
J_uvw = TrialFunction( Q )
u, v, w = split( psi )
