'''
This code solves the Poisson equation Nabla u = f expressed in terms of the function u
The Hessian of u is solved in a post-processing (pp) variational problem, because one cannot take directly the second derivative of u (u.dx(i).dx(j)) [this would lead to divergences]
run with

clear; clear; python3 solve_u.py [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
example:
clear; clear; rm -rf solution; python3 solve_u.py /home/fenics/shared/poisson-equation/mesh /home/fenics/shared/poisson-equation/solution
'''

import colorama as col

from fenics import *
import argparse
from mshr import *
import ufl as ufl
from dolfin import *
import numpy as np
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import input_output as io
import mesh as msh


L = 2.2
h = 0.41

function_space_degree = 4

i, j = ufl.indices( 2 )

parser = argparse.ArgumentParser()
parser.add_argument( "input_directory" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

xdmffile_u = XDMFFile( (args.output_directory) + "/u.xdmf" )

xdmffile_check = XDMFFile( (args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# create mesh
# read the mesh
mesh = Mesh()
xdmf = XDMFFile( mesh.mpi_comm(), (args.input_directory) + "/triangle_mesh.xdmf" )
xdmf.read( mesh )

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

#  norm of vector x
def my_norm(x):
    return (sqrt( np.dot( x, x ) ))


# test for surface elements
dx = Measure( "dx", domain=mesh, subdomain_data=cf, subdomain_id=1 )
ds_l = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=5 )
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds = ds_lr + ds_tb

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

Q = FunctionSpace( mesh, 'P', function_space_degree )
V = VectorFunctionSpace( mesh, 'P', function_space_degree )
T = TensorFunctionSpace( mesh, 'P', function_space_degree, shape=(2, 2) )


class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1 + x[0] ** 2 + 2 * x[1] ** 2
        # values[0] = np.sin( 2 * (np.pi) * (x[0] + x[1]) ) * np.cos( 2 * (np.pi) * (x[0] - x[1]) ** 2 )

    def value_shape(self):
        return (1,)


class grad_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]
        # values[0] = 2 * (np.pi) * cos( 2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * cos( 2 * (np.pi) * ((x[0]) + (x[1])) ) + 4 * (np.pi) * (-(x[0]) + (x[1])) * sin(
        #     2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * sin( 2 * (np.pi) * ((x[0]) + (x[1])) )
        # values[1] = 2 * (np.pi) * cos( 2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * cos( 2 * (np.pi) * ((x[0]) + (x[1])) ) + 4 * (np.pi) * ((x[0]) - (x[1])) * sin(
        #     2 * (np.pi) * ((x[0]) - (x[1])) ** 2 ) * sin( 2 * (np.pi) * ((x[0]) + (x[1])) )

    def value_shape(self):
        return (2,)


class laplacian_u_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 6.0
        # values[0] = 8 * (np.pi) * (-(np.pi) * (1 + 4 * (x[0] - (x[1])) ** 2) * cos( 2 * (np.pi) * (x[0] - (x[1])) ** 2 ) - sin( 2 * (np.pi) * (x[0] - (x[1])) ** 2 )) * sin(
        #     2 * (np.pi) * (x[0] + (x[1])) )

    def value_shape(self):
        return (1,)


class hess_u_exact_expression( UserExpression ):
    def init(self, **kwargs):
        super().init( **kwargs )

    def eval(self, values, x):
        values[0] = 2
        values[1] = 0
        values[2] = 0
        values[3] = 4
        # values[0] = 4 * np.pi * (
        #         4 * np.pi * (-x[0] + x[1]) * np.cos(2 * np.pi * (x[0] + x[1])) * np.sin(2 * np.pi * (x[0] - x[1])**2)
        #         - (np.pi * (1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)) * np.sin(2 * np.pi * (x[0] + x[1]))
        #     )
        # values[1] =  4 * np.pi * (
        #         np.pi * (-1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)
        #     ) * np.sin(2 * np.pi * (x[0] + x[1]))
        # values[2] = 4 * np.pi * (
        #         np.pi * (-1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)
        #     ) * np.sin(2 * np.pi * (x[0] + x[1]))
        # values[3] = 4 * np.pi * (
        #         4 * np.pi * (x[0] - x[1]) * np.cos(2 * np.pi * (x[0] + x[1])) * np.sin(2 * np.pi * (x[0] - x[1])**2)
        #         - (np.pi * (1 + 4 * (x[0] - x[1])**2) * np.cos(2 * np.pi * (x[0] - x[1])**2)
        #         + np.sin(2 * np.pi * (x[0] - x[1])**2)) * np.sin(2 * np.pi * (x[0] + x[1]))
        #     )

    def value_shape(self):
        return (2, 2)


# Define variational problem
u = Function( Q )
nu_u = TestFunction( Q )
f = Function( Q )
grad_u = Function( V )
J_u = TrialFunction( Q )
u_exact = Function( Q )

# Define post-processing (pp) variational problem
# hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
hess_u = Function( T )
nu_hess_u = TestFunction( T )
hess_u_exact = Function( T )
J_hess_u = TrialFunction( T )

u_exact.interpolate( u_exact_expression( element=Q.ufl_element() ) )
grad_u.interpolate( grad_u_expression( element=V.ufl_element() ) )
f.interpolate( laplacian_u_expression( element=Q.ufl_element() ) )

hess_u_exact.interpolate( hess_u_exact_expression( element=T.ufl_element() ) )

bc_u = DirichletBC( Q, u_exact, boundary_tb )

#variational functional for the original problem (poisson equation)
F = (dot( grad( u ), grad( nu_u ) ) + f * nu_u) * dx - dot( n, grad_u ) * nu_u * ds_lr - n[i] * (u.dx( i )) * nu_u * ds_tb

#variational functional for post-processing problem (pp) to obtain the hessian (hess)
F_pp = (hess_u[i, j] * nu_hess_u[i, j] + (u.dx( j )) * ((nu_hess_u[i, j]).dx( i ))) * dx \
       - (n[i] * (u.dx( j )) * nu_hess_u[i, j]) * ds

bcs = [bc_u]
J = derivative( F, u, J_u )
problem = NonlinearVariationalProblem( F, u, bcs, J )
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

J_pp = derivative( F_pp, hess_u, J_hess_u )
problem_pp = NonlinearVariationalProblem( F_pp, hess_u, [], J_pp )
solver_pp = NonlinearVariationalSolver( problem_pp )

#solve original problem
solver.solve()
#solve pp problem
solver_pp.solve()

xdmffile_u.write( u, 0 )
xdmffile_check.write( project( hess_u[i, i], Q ), 0 )
xdmffile_check.write( f, 0 )
xdmffile_check.write( project( hess_u[i, i] - f, Q ), 0 )
xdmffile_check.close()

io.print_scalar_to_csvfile( u, (args.output_directory) + "/u.csv" );


#check if the boundary conditions (BCs) are satisfied
print( "Check of BCs:" )
print( f"\t\t<<(u - phi)^2>>_[partial Omega tb] = {col.Fore.RED}{msh.difference_wrt_measure( u, u_exact, ds_tb ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega lr] = {col.Fore.RED}{msh.difference_wrt_measure( n[i] * (u.dx( i )), n[i] * grad_u[i], ds_lr ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )

print( "Comparison with exact solution: " )
print( f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure( u, u_exact, dx ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"\t\t<<(hess_u - hess_u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure( (hess_u[i, j] - hess_u_exact[i, j]) * (hess_u[i, j] - hess_u_exact[i, j]), Constant( 0 ), dx ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
