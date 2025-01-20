'''
This code solves the biharmonic equation Nabla Nabla u = f expressed in terms of the function u
run with

clear; clear; python3 solve_u.py [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
example:
clear; clear; rm -rf solution; python3 solve_u.py /home/fenics/shared/mesh /home/fenics/shared/solution
'''

from fenics import *
import argparse
from mshr import *
import numpy as np
import ufl as ufl
from dolfin import *

# CHANGE PARAMETERS HERE
L = 2.2
h = 0.41
alpha = 1e3
# CHANGE PARAMETERS HERE


i, j, k, l = ufl.indices( 4 )

parser = argparse.ArgumentParser()
parser.add_argument( "input_directory" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

xdmffile_u = XDMFFile( (args.output_directory) + "/u.xdmf" )

xdmffile_check = XDMFFile( (args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# read the mesh
mesh = Mesh()
xdmf = XDMFFile( mesh.mpi_comm(), (args.input_directory) + "/triangle_mesh.xdmf" )
xdmf.read( mesh )

# radius of the smallest cell in the mesh
r_mesh = mesh.hmin()

print( f"Mesh radius = {r_mesh}" )

# read the triangles
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() )
with XDMFFile( (args.input_directory) + "/triangle_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
cf = cpp.mesh.MeshFunctionSizet( mesh, mvc )
xdmf.close()

# read the lines
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() - 1 )
with XDMFFile( (args.input_directory) + "/line_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
sf = cpp.mesh.MeshFunctionSizet( mesh, mvc )
xdmf.close()

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_lr = 'near(x[0], 0) || near(x[0], 1.0)'
boundary_tb = 'near(x[1], 0) || near(x[1], 1.0)'


# CHANGE PARAMETERS HERE

# read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`

#  norm of vector x
def my_norm(x):
    return (sqrt( np.dot( x, x ) ))


# test for surface elements
dx = Measure( "dx", domain=mesh, subdomain_data=cf, subdomain_id=1 )
ds_l = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=5 )
ds = ds_l + ds_r + ds_t + ds_b

# a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals( UserExpression ):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos( my_norm( np.subtract( x, c_test ) ) - r_test ) ** 2.0

    def value_shape(self):
        return (1,)


f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )

# print out the integrals on the volume and  surface elements and compare them with the exact values to double check that the elements are tagged correctly

exact_value_int_dx = 0.501508
numerical_value_int_dx = assemble( f_test_ds * dx )
print( f"\int f dx = {numerical_value_int_dx}, should be  {exact_value_int_dx}, relative error =  {abs( (numerical_value_int_dx - exact_value_int_dx) / exact_value_int_dx ):e}" )

exact_value_int_ds_l = 0.373168
numerical_value_int_ds_l = assemble( f_test_ds * ds_l )
print( f"\int_l f ds = {numerical_value_int_ds_l}, should be  {exact_value_int_ds_l}, relative error =  {abs( (numerical_value_int_ds_l - exact_value_int_ds_l) / exact_value_int_ds_l ):e}" )

exact_value_int_ds_r = 0.00227783
numerical_value_int_ds_r = assemble( f_test_ds * ds_r )
print( f"\int_r f ds = {numerical_value_int_ds_r}, should be  {exact_value_int_ds_r}, relative error =  {abs( (numerical_value_int_ds_r - exact_value_int_ds_r) / exact_value_int_ds_r ):e}" )

exact_value_int_ds_t = 1.36562
exact_value_int_ds_t = 1.36562
numerical_value_int_ds_t = assemble( f_test_ds * ds_t )
print( f"\int_t f ds = {numerical_value_int_ds_t}, should be  {exact_value_int_ds_t}, relative error =  {abs( (numerical_value_int_ds_t - exact_value_int_ds_t) / exact_value_int_ds_t ):e}" )

exact_value_int_ds_b = 1.02837
numerical_value_int_ds_b = assemble( f_test_ds * ds_b )
print( f"\int_b f ds = {numerical_value_int_ds_b}, should be  {exact_value_int_ds_b}, relative error =  {abs( (numerical_value_int_ds_b - exact_value_int_ds_b) / exact_value_int_ds_b ):e}" )

n = FacetNormal( mesh )

function_space_degree = 3

P_u = FiniteElement( 'P', triangle, function_space_degree )
P_du = VectorElement( 'P', triangle, function_space_degree )
P_ddu = TensorElement( 'P', triangle, function_space_degree, (2, 2) )
P_dddu = TensorElement( 'P', triangle, function_space_degree, (2, 2, 2) )

element = MixedElement( [P_u, P_du, P_ddu, P_ddu ] )
Q = FunctionSpace( mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_du = Q.sub( 1 ).collapse()
Q_ddu = Q.sub( 2 ).collapse()
Q_dddu = Q.sub( 3 ).collapse()


class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1.0 + cos( x[0] ) + sin( x[1] )
    def value_shape(self):
        return (1,)

class grad_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = -sin( x[0] )
        values[1] = cos( x[1] )
    def value_shape(self):
        return (2,)

class hessian_u_expression(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)
    def eval(self, values, x):
        values[0] = cos(2.0 * np.pi * x[0] / L)
        values[1] = cos(2.0 * np.pi * x[1] / h)
        values[2] = sin(2.0 * np.pi * x[0] / L)
        values[3] = sin(2.0 * np.pi * x[1] / h)
        #print("LOCAL tensor\n", values.reshape(self.value_shape()))
    def value_shape(self):
        return (2,2)

class laplacian_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = -cos( x[0] ) - sin( x[1] )
    def value_shape(self):
        return (1,)


# Define variational problem
u = Function( Q_u )
nu = TestFunction( Q_u )
f = Function( Q_u )
grad_u = Function( Q_du )
hessian_u = Function( Q_ddu )
J_u = TrialFunction( Q_u )
u_exact = Function( Q_u )

u_exact.interpolate( u_exact_expression( element=Q_u.ufl_element() ) )
grad_u.interpolate( grad_u_expression( element=Q_du.ufl_element() ) )
hessian_u.interpolate( hessian_u_expression( element=Q_ddu.ufl_element() ) )
f.interpolate( laplacian_u_expression( element=Q_u.ufl_element() ) )

xdmffile_u.write( hessian_u, 0 )



'''

u_profile = Expression( '1.0 + cos(x[0]) + sin(x[1])', L=L, h=h, element=Q_u.ufl_element() )
bc_u = DirichletBC( Q_u, u_profile, boundary )



# \partial_j \partial_j \partial_i \partial_i u = f
# \int dx (\partial_j \partial_j \partial_i \partial_i u ) nu = \int dx f nu
# \int dx \partial_j (\partial_j \partial_i \partial_i u nu ) - \int dx (\partial_j \partial_i \partial_i u) \partial_j nu = \int dx f nu
# 0 =  \int dx (\partial_j \partial_i \partial_i u) \partial_j nu + \int dx f nu  - \int ds n^j [ (\partial_j \partial_i \partial_i u) nu ]

F_u = ((u.dx( i ).dx( i ).dx( j )) * (nu.dx( j )) + f * nu) * dx \
      - n[j] * (u.dx( i ).dx( i ).dx( j )) * nu * ds
# nitsche's term
F_N = alpha / r_mesh * (n[j] * (u.dx( j )) - n[j] * grad_u[j]) * n[k] * (nu.dx( k )) * ds

F = F_u + F_N
bcs = [bc_u]

# u.assign( u_exact )


J = derivative( F, u, J_u )
problem = NonlinearVariationalProblem( F, u, bcs, J )
solver = NonlinearVariationalSolver( problem )

# set the solver parameters here
# params = {'nonlinear_solver': 'newton',
#           'newton_solver':
#               {
#                   'linear_solver': 'mumps',
#                   'absolute_tolerance': 1e-6,
#                   'relative_tolerance': 1e-6,
#                   'maximum_iterations': 1000000,
#                   'relaxation_parameter': 0.95,
#               }
#           }
# solver.parameters.update( params )

solver.solve()

xdmffile_u.write( u, 0 )
xdmffile_check.write( project( u.dx( i ).dx( i ).dx( j ).dx( j ), Q_u ), 0 )
xdmffile_check.write( f, 0 )
xdmffile_check.write( project( u.dx( i ).dx( i ).dx( j ).dx( j ) - f, Q_u ), 0 )
xdmffile_check.close()


def errornorm(u_e, u):
    error = (u_e - u) ** 2 * dx
    E = sqrt( abs( assemble( error ) ) )
    V = u.function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    W = FunctionSpace( mesh, 'P', degree + 3 )
    u_e_W = interpolate( u_e, W )
    u_W = interpolate( u, W )
    e_W = Function( W )
    e_W.vector()[:] = u_e_W.vector().get_local() - u_W.vector().get_local()
    error = e_W ** 2 * dx
    return sqrt( abs( assemble( error ) ) )


print( "Solution check: " )
print( f"\t<<(u - u_exact)^2>>_no-errornorm = {assemble( ((u - u_exact) ** 2) * dx ) / assemble( Constant( 1.0 ) * dx )}" )
print( f"\t<<(u - u_exact)^2>>_errornorm = {errornorm( u, u_exact )}" )

print( f"\t<<(Nabla u - f)^2>>_no-errornorm = {assemble( ((u.dx( i ).dx( i ).dx( j ).dx( j ) - f) ** 2) * dx ) / assemble( Constant( 1.0 ) * dx )}" )
print( f"\t<<(Nabla u - f)^2>>_errornorm = {errornorm( project( u.dx( i ).dx( i ).dx( j ).dx( j ), Q_u ), f )}" )

print( f"\t<<(n[i] \partial_i u - n[i] grad_u[i])^2>> =  {assemble( ((n[i] * grad_u[i]) - (n[i] * (u.dx( i )))) ** 2 * ds ) / assemble( Constant( 1.0 ) * ds )}" )
'''