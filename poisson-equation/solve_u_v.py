'''
This code solves the biharmonic equation Nabla Nabla u = f expressed in terms of the function u and v = Nabla u
run with

clear; clear; python3 solve_u_v.py [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
example:
clear; clear; rm -rf solution; python3 solve_u_v.py /home/fenics/shared/poisson-equation/mesh /home/fenics/shared/poisson-equation/solution
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
alpha = 1e2
function_space_degree = 4
# CHANGE PARAMETERS HERE

i, j, k, l = ufl.indices( 4 )

parser = argparse.ArgumentParser()
parser.add_argument( "input_directory" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

xdmffile_u = XDMFFile( (args.output_directory) + "/u.xdmf" )
xdmffile_u.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

xdmffile_v = XDMFFile( (args.output_directory) + "/v.xdmf" )
xdmffile_v.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

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
boundary_lr = 'near(x[0], 0) || near(x[0], 2.2)'
boundary_tb = 'near(x[1], 0) || near(x[1], 0.41)'
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
numerical_value_int_ds_t = assemble( f_test_ds * ds_t )
print( f"\int_t f ds = {numerical_value_int_ds_t}, should be  {exact_value_int_ds_t}, relative error =  {abs( (numerical_value_int_ds_t - exact_value_int_ds_t) / exact_value_int_ds_t ):e}" )

exact_value_int_ds_b = 1.02837
numerical_value_int_ds_b = assemble( f_test_ds * ds_b )
print( f"\int_b f ds = {numerical_value_int_ds_b}, should be  {exact_value_int_ds_b}, relative error =  {abs( (numerical_value_int_ds_b - exact_value_int_ds_b) / exact_value_int_ds_b ):e}" )

n = FacetNormal( mesh )


P_u = FiniteElement( 'P', triangle, function_space_degree )
P_v = VectorElement( 'P', triangle, function_space_degree )
element = MixedElement( [P_u, P_v] )
Q = FunctionSpace( mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_v = Q.sub( 1 ).collapse()


class u_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = np.sin(2 * (np.pi) * (x[0] + x[1])) *  np.cos(2 * (np.pi) * (x[0] - x[1])**2)
    def value_shape(self):
        return (1,)

class v_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] =  2 *(np.pi) *cos(2 *(np.pi) *((x[0]) - (x[1]))**2) * cos(2 *(np.pi) *((x[0]) + (x[1]))) + 4 *(np.pi) *(-(x[0]) + (x[1]))* sin(2 *(np.pi) * ((x[0]) - (x[1]))**2) * sin(2 * (np.pi) * ((x[0]) + (x[1])))
        values[1] = 2 * (np.pi) * cos(2* (np.pi) * ((x[0]) - (x[1]))**2) * cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4* (np.pi) * ((x[0]) - (x[1])) * sin(2 *(np.pi) *((x[0]) - (x[1]))**2) * sin(2 * (np.pi)*  ((x[0]) + (x[1])))
    def value_shape(self):
        return (2,)

class laplacian_u_exact_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 8 *(np.pi)* (-(np.pi)* (1+4* (x[0]-(x[1]))**2) * cos(2* (np.pi)* (x[0]-(x[1]))**2)-sin(2* (np.pi) *(x[0]-(x[1]))**2))* sin(2* (np.pi)* (x[0]+(x[1])))
    def value_shape(self):
        return (1,)



# Define variational problem
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

u_exact.interpolate( u_exact_expression( element=Q_u.ufl_element() ) )
v_exact.interpolate( v_exact_expression( element=Q_v.ufl_element() ) )
laplacian_u_exact.interpolate( laplacian_u_exact_expression( element=Q_u.ufl_element() ) )
f.interpolate( laplacian_u_exact_expression( element=Q_u.ufl_element() ) )

u_profile = Expression( 'sin(2.0*pi*(x[0]+x[1])) * cos(2.0*pi*pow(x[0]-x[1], 2))', L=L, h=h, element=Q.sub( 0 ).ufl_element() )
bc_u = DirichletBC( Q.sub( 0 ), u_profile, boundary )

F_v = (v[i] * nu_v[i] + u * (nu_v[i].dx(i))) * dx \
      - n[i] * u * (nu_v[i]) * ds
F_u = (v[i] * (nu_u.dx( i )) + f * nu_u) * dx \
      - n[i] * v[i] * nu_u * ds
F_N = alpha/r_mesh * (n[i] * v[i] - n[i] * v_exact[i]) * n[j] * nu_v[j] * ds


F = F_u + F_v + F_N
bcs = [bc_u]

J = derivative( F, psi, J_uv )
problem = NonlinearVariationalProblem( F, psi, bcs, J )
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

u_output, v_output = psi.split( deepcopy=True )

xdmffile_u.write( u_output, 0 )
xdmffile_v.write( v_output, 0 )

print( "BCs check: " )
print( f"\t<<(u-u_exact)^2>>_\partial Omega =  {assemble( (u_output - u_exact) ** 2 * ds ) / assemble( Constant( 1.0 ) * ds )}" )
print( f"\t<<(n.v-n.v_exact)^2>>_\partial Omega =  {assemble( (n[i]*v_output[i] - n[i]*v_exact[i]) ** 2 * ds ) / assemble( Constant( 1.0 ) * ds )}" )
#
print( "Solution check: " )
print( f"\t<<(u - u_exact)^2>> = {sqrt( assemble( ((u_output - u_exact) ** 2) * dx ) / assemble( Constant( 1.0 ) * dx ) )}" )
print( f"\t<<(v - v_exact)^2>> = {sqrt( assemble( ((v_output[i] - v_exact[i]) * (v_output[i] - v_exact[i])) * dx ) / assemble( Constant( 1.0 ) * dx ) )}" )

'''
xdmffile_u.write( v_output, 0 )
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
    return sqrt( abs( assemble( error ) / assemble( Constant( 1.0 ) * dx ) ) )

print( f"\t<<(Nabla u - f)^2>>_no-errornorm = {assemble( ((u.dx( i ).dx( i ).dx( j ).dx( j ) - f) ** 2) * dx ) / assemble( Constant( 1.0 ) * dx )}" )
print( f"\t<<(Nabla u - f)^2>>_errornorm = {errornorm( project( u.dx( i ).dx( i ).dx( j ).dx( j ), Q_u ), f )}" )
'''