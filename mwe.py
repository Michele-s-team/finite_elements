'''
This code solves the biharmonic equation Nabla Nabla u = f expressed in terms of the function u
run with

clear; clear; python3 mwe.py
example:
python3 mwe.py
'''

from fenics import *
from mshr import *
import ufl as ufl
from dolfin import *

L = 2.2
h = 0.41

i, j = ufl.indices( 2 )


xdmffile_u = XDMFFile( "solution/u.xdmf" )
xdmffile_check = XDMFFile( "solution/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# create mesh
mesh = RectangleMesh(Point(0, 0), Point(L, h), 20, 10)
mf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
eps = 100 * DOLFIN_EPS


class left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0, eps)
left().mark(mf, 2)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L, eps)
right().mark(mf, 3)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], h, eps)
top().mark(mf, 4)

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0, eps)
bottom().mark(mf, 5)

boundary = 'on_boundary'

# test for surface elements
ds_l = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)


n = FacetNormal( mesh )

function_space_degree = 4
Q = FunctionSpace( mesh, 'P', function_space_degree )
V = VectorFunctionSpace( mesh, 'P', function_space_degree )


class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1.0 + (x[0] ** 4) - 2.0 * (x[1] ** 4)
    def value_shape(self):
        return (1,)

class grad_laplacian_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 24.0 * x[0]
        values[1] = - 48.0 * x[1]
    def value_shape(self):
        return (2,)

class laplacian2_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = -24.0
    def value_shape(self):
        return (1,)


u = Function( Q )
nu = TestFunction( Q )
f = Function( Q )
grad_laplacian_u = Function( V )
J_u = TrialFunction( Q )
u_exact = Function( Q )

u_exact.interpolate( u_exact_expression( element=Q.ufl_element() ) )
grad_laplacian_u.interpolate( grad_laplacian_u_expression( element=V.ufl_element() ) )
f.interpolate( laplacian2_u_expression( element=Q.ufl_element() ) )

u_profile = Expression( '1.0 + pow(x[0], 4) - 2.0 * pow(x[1], 4)', L=L, h=h, element=Q.ufl_element() )
bc_u = DirichletBC( Q, u_profile, boundary )

F = ((u.dx( i ).dx( i ).dx( j )) * (nu.dx( j )) + f * nu) * dx \
    - n[j] * grad_laplacian_u[j] * nu * (ds_l + ds_r + ds_t + ds_b)
bcs = [bc_u]

J = derivative( F, u, J_u )
problem = NonlinearVariationalProblem( F, u, bcs, J )
solver = NonlinearVariationalSolver( problem )

solver.solve()

xdmffile_u.write( u, 0 )
xdmffile_check.write( project( u.dx( i ).dx( i ).dx( j ).dx( j ), Q ), 0 )
xdmffile_check.write( f, 0 )
xdmffile_check.write( project(  u.dx( i ).dx( i ).dx( j ).dx( j ) - f, Q ), 0 )
xdmffile_check.close()