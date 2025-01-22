from fenics import *
from mshr import *
import ufl as ufl
from dolfin import *
import numpy as np

L = 1.0
h = 1.0
function_space_degree = 2

i, j = ufl.indices( 2 )


# create mesh
mesh = RectangleMesh( Point( 0, 0 ), Point( L, h ), 20, 20 )
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() )
cf = cpp.mesh.MeshFunctionSizet( mesh, mvc )
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() - 1 )
mf = cpp.mesh.MeshFunctionSizet( mesh, mvc )

boundary = 'on_boundary'

dx = Measure( "dx", domain=mesh, subdomain_data=cf)
ds = Measure( "ds", domain=mesh, subdomain_data=mf )


n = FacetNormal( mesh )

P_u = FiniteElement( 'P', triangle, function_space_degree )
P_v = FiniteElement( 'P', triangle, function_space_degree )
P_w = FiniteElement( 'P', triangle, function_space_degree )
element = MixedElement( [P_u, P_v, P_w] )
Q = FunctionSpace( mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_v = Q.sub( 1 ).collapse()
Q_w = Q.sub( 2 ).collapse()


class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = cos( x[0] + x[1] ) * sin( x[0] - x[1] )

    def value_shape(self):
        return (1,)


class v_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = - 4 * cos( x[0] ) * sin( x[0] ) + 4 * cos( x[1] ) * sin( x[1] )

    def value_shape(self):
        return (1,)


class w_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 8 * (sin( 2 * x[0] ) - sin( 2 * x[1] ))

    def value_shape(self):
        return (1,)


psi = Function( Q )
nu_u, nu_v, nu_w = TestFunctions( Q )

u_output = Function( Q_u )
v_output = Function( Q_v )
w_output = Function( Q_w )


f = Function( Q_w )
J_uvw = TrialFunction( Q )
u, v, w = split( psi )

f.interpolate( w_exact_expression( element=Q_w.ufl_element() ) )

u_profile = Expression( 'cos(x[0]+x[1]) * sin(x[0]-x[1])', L=L, h=h, element=Q.sub( 0 ).ufl_element() )
v_profile = Expression( '- 4 * cos(x[0])*sin(x[0]) + 4 * cos(x[1])*sin(x[1])', L=L, h=h, element=Q.sub( 1 ).ufl_element() )
bc_u = DirichletBC( Q.sub( 0 ), u_profile, boundary )
bc_v = DirichletBC( Q.sub( 1 ), v_profile, boundary )

F_v = ((v.dx( i )) * (nu_u.dx( i )) + f * nu_u) * dx  - n[i] * (v.dx( i )) * nu_u * ds
F_u = ((u.dx( i )) * (nu_v.dx( i )) + v * nu_v) * dx  - n[i] * (u.dx( i )) * nu_v * ds
F_w = ((v.dx( i )) * (nu_w.dx( i )) + w * nu_w) * dx - n[i] * (v.dx( i )) * nu_w * ds

F = F_u + F_v + F_w
bcs = [bc_u, bc_v]

J = derivative( F, psi, J_uvw )
problem = NonlinearVariationalProblem( F, psi, bcs, J )
solver = NonlinearVariationalSolver( problem )

solver.solve()

u_output, v_output, w_output = psi.split( deepcopy=True )

print( "Check that the PDE is satisfied: " )
print( f"\t<<(w-f)^2>>_Omega =  {assemble( (w_output - f) ** 2 * dx ) / assemble( Constant( 1.0 ) * dx )}" )
print( f"\t<<(w-f)^2>>_[boundary of Omega] =  {assemble( (w_output - f) ** 2 * ds ) / assemble( Constant( 1.0 ) * ds )}" )

xdmffile_check = XDMFFile( "check.xdmf" )
xdmffile_check.write( project( w_output - f , Q_w ), 0 )
xdmffile_check.close()