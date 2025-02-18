'''
This code solves the Poisson equation  Nabla u = f expressed in terms of the function u and v_i = \partial_i u
run with

clear; clear; python3 solve_u_v.py [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
example:
clear; clear; rm -rf solution; python3 solve_u_v.py /home/fenics/shared/poisson-equation/mesh /home/fenics/shared/poisson-equation/solution
'''

from fenics import *
import argparse
import colorama as col
from mshr import *
import numpy as np
import ufl as ufl
from dolfin import *
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )


import input_output as io
import mesh as msh


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
msh.test_mesh_integral(0.501508, f_test_ds, dx, '\int f dx')

msh.test_mesh_integral(0.373168, f_test_ds, ds_l, '\int_l f ds')
msh.test_mesh_integral(0.00227783, f_test_ds, ds_r, '\int_r f ds')
msh.test_mesh_integral(1.36562, f_test_ds, ds_t, '\int_t f ds')
msh.test_mesh_integral(1.02837, f_test_ds, ds_b, '\int_b f ds')


n = FacetNormal( mesh )

#define elements and function spaces
P_u = FiniteElement( 'P', triangle, function_space_degree )
P_v = VectorElement( 'P', triangle, function_space_degree )
element = MixedElement( [P_u, P_v] )
Q = FunctionSpace( mesh, element )

Q_u = Q.sub( 0 ).collapse()
Q_v = Q.sub( 1 ).collapse()

#define exact solution to set the boundary conditions (BCs) and compare the finite-element (FE) solution with the exact one
# CHANGE PARAMETERS HERE
class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = 1 + cos( x[0] - x[1] ) - sin( x[1] )
        # values[0] = 1 + (x[0]**2) + 2 * (x[1]**2)
        values[0] = np.sin(2 * (np.pi) * (x[0] + x[1])) *  np.cos(2 * (np.pi) * (x[0] - x[1])**2)

    def value_shape(self):
        return (1,)


class v_exact_expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = - sin( x[0] - x[1] )
        # values[1] = sin( x[0] - x[1] ) - cos( x[1] )
        values[0] =  2 *(np.pi) *cos(2 *(np.pi) *((x[0]) - (x[1]))**2) * cos(2 *(np.pi) *((x[0]) + (x[1]))) + 4 *(np.pi) *(-(x[0]) + (x[1]))* sin(2 *(np.pi) * ((x[0]) - (x[1]))**2) * sin(2 * (np.pi) * ((x[0]) + (x[1])))
        values[1] = 2 * (np.pi) * cos(2* (np.pi) * ((x[0]) - (x[1]))**2) * cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4* (np.pi) * ((x[0]) - (x[1])) * sin(2 *(np.pi) *((x[0]) - (x[1]))**2) * sin(2 * (np.pi)*  ((x[0]) + (x[1])))

    def value_shape(self):
        return (2,)


class laplacian_u_exact_expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = -2 * cos( x[0] - x[1] ) + sin( x[1] )
        values[0] = 8 *(np.pi)* (-(np.pi)* (1+4* (x[0]-(x[1]))**2) * cos(2* (np.pi)* (x[0]-(x[1]))**2)-sin(2* (np.pi) *(x[0]-(x[1]))**2))* sin(2* (np.pi)* (x[0]+(x[1])))

    def value_shape(self):
        return (1,)
# CHANGE PARAMETERS HERE


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

u_exact.interpolate( u_exact_expression( element=Q_u.ufl_element() ) )
v_exact.interpolate( v_exact_expression( element=Q_v.ufl_element() ) )
laplacian_u_exact.interpolate( laplacian_u_exact_expression( element=Q_u.ufl_element() ) )
f.interpolate( laplacian_u_exact_expression( element=Q_u.ufl_element() ) )

#define Difichlet boundary conditions
bc_u = DirichletBC( Q.sub( 0 ), u_exact, boundary )

#define variational problem
F_v = (v[i] * nu_v[i] + u * (nu_v[i].dx( i ))) * dx \
      - n[i] * u * (nu_v[i]) * ds
F_u = (v[i] * (nu_u.dx( i )) + f * nu_u) * dx \
      - n[i] * v[i] * nu_u * ds
F_N = alpha / r_mesh * (n[i] * v[i] - n[i] * v_exact[i]) * n[j] * nu_v[j] * ds

F = F_u + F_v + F_N
bcs = [bc_u]

J = derivative( F, psi, J_uv )
problem = NonlinearVariationalProblem( F, psi, bcs, J )
solver = NonlinearVariationalSolver( problem )
# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update( params )

#solve the variational problem
solver.solve()

#print out the solution
u_output, v_output = psi.split( deepcopy=True )

xdmffile_u.write( u_output, 0 )
xdmffile_v.write( v_output, 0 )

io.print_scalar_to_csvfile( u_output, (args.output_directory) + "/u.csv" );
io.print_vector_to_csvfile( v_output, (args.output_directory) + "/v.csv" );

#check if the boundary conditions are satisfied
print( "BCs check: " )
print( f"\t\t<<(u - u_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure( u_output, u_exact, ds ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(n.v-n.v_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(  n[i] * v_output[i] , n[i] * v_exact[i], ds ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )

#check if the FE solution agrees with the exact one
print( "Comparison with exact solution: " )
print( f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure( u_output, u_exact, dx ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<|v - v_exact|^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure( sqrt((v_output[i] - v_exact[i]) * (v_output[i] - v_exact[i])), Constant(0), dx ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


