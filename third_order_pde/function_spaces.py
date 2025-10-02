from fenics import *

import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import mesh.load as lmsh
import parameters.read.solution as rpam

element_geometry = msh.element_geometry(lmsh.mesh)


P_u = FiniteElement( 'P', element_geometry, rpam.parameters['function_space_degree'] )
P_v = FiniteElement( 'P', element_geometry, rpam.parameters['function_space_degree'] )
element = MixedElement( [P_u, P_v] )

# mixed function space for main variational problem
Q = FunctionSpace( lmsh.mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_v = Q.sub( 1 ).collapse()



psi = Function( Q )
nu_u, nu_v = TestFunctions( Q )

u_output = Function( Q_u )
v_output = Function( Q_v )

u_exact = Function( Q_u )
v_exact = Function( Q_v )



f = Function( Q_u )
J_Q = TrialFunction( Q )

u, v = split( psi )
