from fenics import *
from mshr import *
import ufl as ufl
from dolfin import *
import numpy as np
import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import geometry as geo
import mesh as msh




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

eps = 100 * DOLFIN_EPS


class left( SubDomain ):
    def inside(self, x, on_boundary):
        return near( x[0], 0, eps )


left().mark( mf, 2 )


class right( SubDomain ):
    def inside(self, x, on_boundary):
        return near( x[0], L, eps )


right().mark( mf, 3 )


class top( SubDomain ):
    def inside(self, x, on_boundary):
        return near( x[1], h, eps )


top().mark( mf, 4 )


class bottom( SubDomain ):
    def inside(self, x, on_boundary):
        return near( x[1], 0, eps )


bottom().mark( mf, 5 )

#Nt^i_notes on \partial \Omega_in and out
def n_lr():
    x = ufl.SpatialCoordinate(mesh)
    return( as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0] ))

def n_tb():
    x = ufl.SpatialCoordinate(mesh)
    return( as_tensor([0, conditional(lt(x[1], h/2.0), -1.0, 1.0)] ))

boundary = 'on_boundary'

dx = Measure( "dx", domain=mesh, subdomain_data=cf )
ds = Measure( "ds", domain=mesh, subdomain_data=mf )
# test for surface elements
ds_l = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=5 )

# a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals( UserExpression ):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos( geo.my_norm( np.subtract( x, c_test ) ) - r_test ) ** 2.0

    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )


msh.test_mesh_integral(0.937644045077037, f_test_ds, dx, '\int f dx')
msh.test_mesh_integral(0.962047, f_test_ds, ds_l, '\int_l f ds')
msh.test_mesh_integral(0.805631, f_test_ds, ds_r, '\int_r f ds')
msh.test_mesh_integral(0.975624, f_test_ds, ds_t, '\int_t f ds')
msh.test_mesh_integral(0.776577, f_test_ds, ds_b, '\int_b f ds')




n = FacetNormal( mesh )

P_u = FiniteElement( 'P', triangle, function_space_degree )
P_v = FiniteElement( 'P', triangle, function_space_degree )
P_w = FiniteElement( 'P', triangle, function_space_degree )
element = MixedElement( [P_u, P_v, P_w] )
Q = FunctionSpace( mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_v = Q.sub( 1 ).collapse()
Q_w = Q.sub( 2 ).collapse()
Q_grad_v = VectorFunctionSpace( mesh, 'P', function_space_degree )


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


class grad_v_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = -4 * cos( 2 * x[0] )
        values[1] = 4 * cos( 2 * x[1] )

    def value_shape(self):
        return (2,)


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
grad_v_exact = Function( Q_grad_v )
J_uvw = TrialFunction( Q )
u, v, w = split( psi )

f.interpolate( w_exact_expression( element=Q_w.ufl_element() ) )
grad_v_exact.interpolate( grad_v_exact_expression( element=Q_grad_v.ufl_element() ) )

u_profile = Expression( 'cos(x[0]+x[1]) * sin(x[0]-x[1])', L=L, h=h, element=Q.sub( 0 ).ufl_element() )
v_profile = Expression( '- 4 * cos(x[0])*sin(x[0]) + 4 * cos(x[1])*sin(x[1])', L=L, h=h, element=Q.sub( 1 ).ufl_element() )
w_profile = Expression( '8 * (sin( 2 * x[0] ) - sin( 2 * x[1] ))', L=L, h=h, element=Q.sub( 2 ).ufl_element() )
bc_u = DirichletBC( Q.sub( 0 ), u_profile, boundary )
bc_v = DirichletBC( Q.sub( 1 ), v_profile, boundary )
bc_w = DirichletBC( Q.sub( 2 ), w_profile, boundary )

F_v = ((v.dx( i )) * (nu_u.dx( i )) + f * nu_u) * dx
F_u = ((u.dx( i )) * (nu_v.dx( i )) + v * nu_v) * dx
F_w = ((v.dx( i )) * (nu_w.dx( i )) + w * nu_w) * dx - n[i] * grad_v_exact[i] * nu_w * ds

F = F_u + F_v + F_w
# bcs = [bc_u, bc_v]
bcs = [bc_u, bc_v, bc_w]

J = derivative( F, psi, J_uvw )
problem = NonlinearVariationalProblem( F, psi, bcs, J )
solver = NonlinearVariationalSolver( problem )

params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  # 'linear_solver'           : 'superlu',
                  'linear_solver': 'mumps',
                  'absolute_tolerance': 1e-12,
                  'relative_tolerance': 1e-12,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update( params )

solver.solve()

u_output, v_output, w_output = psi.split( deepcopy=True )

print( "Check that the PDE is satisfied: " )
print( f"\t<<(w-f)^2>>_Omega =  {assemble( (w_output - f) ** 2 * dx ) / assemble( Constant( 1.0 ) * dx )}" )
print( f"\t<<(w-f)^2>>_[bulk of Omega ] =  {msh.difference_in_bulk(w_output, f)}" )


print( f"\t<<(w-f)^2>>_[boundary of Omega] =  {assemble( (w_output - f) ** 2 * ds ) / assemble( Constant( 1.0 ) * ds )}" )

xdmffile_check = XDMFFile( "check.xdmf" )
xdmffile_check.write( project( w_output - f, Q_w ), 0 )
xdmffile_check.close()
