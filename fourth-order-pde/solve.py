'''
This code solves the biharmonic equation Nabla Nabla \partial_i (z \partial_i z) = f expressed in terms of the function
- z
- omega[i] = \partial_i z
- mu = \partial_i (z omega_i)
- rho_i = \partial_i mu
- tau = \partial_i rho_i
run with

clear; clear; python3 solve.py [path where to read the mesh generated from generate_square_mesh.py or generate_ring_mesh.py] [path where to store the solution]
example:
clear; clear; rm -rf solution; python3 solve.py /home/fenics/shared/fourth-order-pde/mesh /home/fenics/shared/fourth-order-pde/solution
'''

from fenics import *
import argparse

from mshr import *
import ufl as ufl
from dolfin import *
import termcolor
import numpy as np
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import input_output as io
import geometry as geo
import mesh as msh

alpha = 1e2
'''
#square mesh
# CHANGE PARAMETERS HERE
L = 1.0
h = 1.0
# CHANGE PARAMETERS HERE
'''
#
# ring mesh
# CHANGE PARAMETERS HERE
r = 1.0
R = 2.0
# CHANGE PARAMETERS HERE
#

# CHANGE PARAMETERS HERE
function_space_degree = 4
# CHANGE PARAMETERS HERE


i, j, k, l = ufl.indices( 4 )

parser = argparse.ArgumentParser()
parser.add_argument( "input_directory" )
parser.add_argument( "output_directory" )
args = parser.parse_args()

xdmffile_z = XDMFFile( (args.output_directory) + "/z.xdmf" )
xdmffile_omega = XDMFFile( (args.output_directory) + "/omega.xdmf" )
xdmffile_mu = XDMFFile( (args.output_directory) + "/mu.xdmf" )
xdmffile_rho = XDMFFile( (args.output_directory) + "/rho.xdmf" )
xdmffile_tau = XDMFFile( (args.output_directory) + "/tau.xdmf" )

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
cf = dolfin.cpp.mesh.MeshFunctionSizet( mesh, mvc )
xdmf.close()

# read the lines
mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() - 1 )
with XDMFFile( (args.input_directory) + "/line_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )
sf = dolfin.cpp.mesh.MeshFunctionSizet( mesh, mvc )
xdmf.close()

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
# CHANGE PARAMETERS HERE


# test for surface elements
# square mesh
'''
dx = Measure( "dx", domain=mesh, subdomain_data=cf, subdomain_id=1 )
ds_l = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=5 )
ds = ds_l + ds_r + ds_t + ds_b
'''

# ring mesh
#
dx = Measure( "dx", domain=mesh, subdomain_data=cf, subdomain_id=1 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
ds_R = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )
ds = ds_r + ds_R
#

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

# print out the integrals on the volume and  surface elements and compare them with the exact values to double check that the elements are tagged correctly


# square mesh
'''
exact_value_int_dx = 0.937644
numerical_value_int_dx = assemble( f_test_ds * dx )
print( f"\int f dx = {numerical_value_int_dx}, should be  {exact_value_int_dx}, relative error =  {abs( (numerical_value_int_dx - exact_value_int_dx) / exact_value_int_dx ):e}" )

exact_value_int_ds_l = 0.962047
numerical_value_int_ds_l = assemble( f_test_ds * ds_l )
print( f"\int_l f ds = {numerical_value_int_ds_l}, should be  {exact_value_int_ds_l}, relative error =  {abs( (numerical_value_int_ds_l - exact_value_int_ds_l) / exact_value_int_ds_l ):e}" )

exact_value_int_ds_r = 0.805631
numerical_value_int_ds_r = assemble( f_test_ds * ds_r )
print( f"\int_r f ds = {numerical_value_int_ds_r}, should be  {exact_value_int_ds_r}, relative error =  {abs( (numerical_value_int_ds_r - exact_value_int_ds_r) / exact_value_int_ds_r ):e}" )

exact_value_int_ds_t = 0.975624
numerical_value_int_ds_t = assemble( f_test_ds * ds_t )
print( f"\int_t f ds = {numerical_value_int_ds_t}, should be  {exact_value_int_ds_t}, relative error =  {abs( (numerical_value_int_ds_t - exact_value_int_ds_t) / exact_value_int_ds_t ):e}" )

exact_value_int_ds_b = 0.776577
numerical_value_int_ds_b = assemble( f_test_ds * ds_b )
print( f"\int_b f ds = {numerical_value_int_ds_b}, should be  {exact_value_int_ds_b}, relative error =  {abs( (numerical_value_int_ds_b - exact_value_int_ds_b) / exact_value_int_ds_b ):e}" )
'''

# ring mesh
#
msh.test_mesh_integral( 2.90212, f_test_ds, dx, '\int f dx' )
msh.test_mesh_integral( 2.77595, f_test_ds, ds_r, '\int_r f ds' )
msh.test_mesh_integral( 3.67175, f_test_ds, ds_R, '\int_R f ds' )
#

n = FacetNormal( mesh )

P_z = FiniteElement( 'P', triangle, function_space_degree )
P_omega = VectorElement( 'P', triangle, function_space_degree )
P_mu = FiniteElement( 'P', triangle, function_space_degree )
P_rho = VectorElement( 'P', triangle, function_space_degree )
P_tau = FiniteElement( 'P', triangle, function_space_degree )
element = MixedElement( [P_z, P_omega, P_mu, P_rho, P_tau] )
Q = FunctionSpace( mesh, element )

Q_z = Q.sub( 0 ).collapse()
Q_omega = Q.sub( 1 ).collapse()
Q_mu = Q.sub( 2 ).collapse()
Q_rho = Q.sub( 3 ).collapse()
Q_tau = Q.sub( 4 ).collapse()

assigner = FunctionAssigner( Q, [Q_z, Q_omega, Q_mu, Q_rho, Q_tau] )


class z_exact_expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = np.cos( x[0] + x[1] ) * np.sin( x[0] - x[1] )
        values[0] = (x[0] ** 4 + x[1] ** 4) / 48.0

    def value_shape(self):
        return (1,)


class omega_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = (x[0] ** 3) / 12.0
        values[1] = (x[1] ** 3) / 12.0

    def value_shape(self):
        return (2,)


class mu_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = (7 * x[0] ** 6 + 3 * x[0] ** 4 * x[1] ** 2 + 3 * x[0] ** 2 * x[1] ** 4 + 7 * x[1] ** 6) / 576.0

    def value_shape(self):
        return (1,)


class rho_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = x[0] * (7 * x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + x[1] ** 4) / 96.0
        values[1] = x[1] * (x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + 7 * x[1] ** 4) / 96.0

    def value_shape(self):
        return (2,)


class f_exact_expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = -16 * (np.cos( 4 * x[0] ) + np.cos( 4 * x[1] ) + np.sin( 2 * x[0] ) * np.sin( 2 * x[1] ))
        values[0] = 1 / 8.0 * (3 * x[0] ** 4 + x[0] ** 2 * x[1] ** 2 + 3 * x[1] ** 4)

    def value_shape(self):
        return (1,)


# Define variational problem
psi = Function( Q )
nu_z, nu_omega, nu_mu, nu_rho, nu_tau = TestFunctions( Q )

z_output = Function( Q_z )
omega_output = Function( Q_omega )
mu_output = Function( Q_mu )
rho_output = Function( Q_rho )
tau_output = Function( Q_tau )

z_exact = Function( Q_z )
omega_exact = Function( Q_omega )
mu_exact = Function( Q_mu )
rho_exact = Function( Q_rho )
tau_exact = Function( Q_tau )

f = Function( Q_z )
J_Q = TrialFunction( Q )
z, omega, mu, rho, tau = split( psi )

z_exact.interpolate( z_exact_expression( element=Q_z.ufl_element() ) )
omega_exact.interpolate( omega_exact_expression( element=Q_omega.ufl_element() ) )
mu_exact.interpolate( mu_exact_expression( element=Q_mu.ufl_element() ) )
rho_exact.interpolate( rho_exact_expression( element=Q_rho.ufl_element() ) )
tau_exact.interpolate( f_exact_expression( element=Q_tau.ufl_element() ) )
f.interpolate( f_exact_expression( element=Q_z.ufl_element() ) )

z_profile = Expression( '(pow(x[0], 4) + pow(x[1], 4)) / 48.0', element=Q.sub( 0 ).ufl_element() )
mu_profile = Expression( '(7 * pow(x[0], 6) + 3 * pow(x[0], 4) * pow(x[1], 2) + 3 * pow(x[0], 2) * pow(x[1], 4) + 7 * pow(x[1], 6))/576.0', element=Q.sub( 2 ).ufl_element() )
rho_profile = Expression(
    ('(1.0 / 96.0) * x[0] * (7.0 * pow(x[0], 4) + 2.0 * pow(x[0], 2) * pow(x[1], 2) + pow(x[1], 4))', '(1.0 / 96.0) * x[1] * (pow(x[0], 4) + 2 * pow(x[0], 2) * pow(x[1], 2) + 7 * pow(x[1], 4))'),
    element=Q.sub( 3 ).ufl_element() )
tau_profile = Expression( '(1.0 / 8.0) * (3 * pow(x[0], 4) + pow(x[0], 2) * pow(x[1], 2) + 3 * pow(x[1], 4))', element=Q.sub( 4 ).ufl_element() )

bc_z = DirichletBC( Q.sub( 0 ), z_profile, boundary )
bc_mu = DirichletBC( Q.sub( 2 ), mu_profile, boundary )
bc_rho = DirichletBC( Q.sub( 3 ), rho_profile, boundary )
bc_tau = DirichletBC( Q.sub( 4 ), tau_profile, boundary )

# here is assign a wrong value to u (f) on purpose to see whether the solver conveges to the right solution
assigner.assign( psi, [f, omega_exact, mu_exact, rho_exact, tau_exact] )

F_z = ((mu.dx( j )) * (nu_z.dx( j )) + f * nu_z) * dx \
      - n[j] * (mu.dx( j )) * nu_z * ds

F_omega = (z * ((nu_omega[i]).dx( i )) + omega[i] * nu_omega[i]) * dx \
          - n[i] * z * nu_omega[i] * ds

# F_mu = ((z * omega[i]).dx(i) * nu_mu  - mu * nu_mu) * dx
F_mu = (z * omega[i] * (nu_mu.dx( i )) + mu * nu_mu) * dx \
       - n[i] * z * omega[i] * nu_mu * ds

F_rho = (mu * ((nu_rho[i]).dx( i )) + rho[i] * nu_rho[i]) * dx \
        - n[i] * mu * nu_rho[i] * ds

F_tau = (tau * nu_tau + rho[i] * (nu_tau.dx( i ))) * dx \
        - n[i] * rho[i] * nu_tau * ds

F_N = alpha / r_mesh * ( \
            (n[i] * omega[i] - n[i] * omega_exact[i]) * n[j] * nu_omega[j] * ds \
            + (mu - ((z * omega[i]).dx( i ))) * nu_mu * ds \
    )

F = (F_omega + F_z + F_mu + F_rho + F_tau) + F_N
# bcs = [bc_z]
# bcs = [bc_z, bc_mu, bc_rho, bc_tau]
bcs = [bc_z, bc_rho, bc_tau]

J = derivative( F, psi, J_Q )
problem = NonlinearVariationalProblem( F, psi, bcs, J )
solver = NonlinearVariationalSolver( problem )
# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  # 'linear_solver': 'superlu',
                  'linear_solver': 'mumps',
                  'absolute_tolerance': 1e-6,
                  'relative_tolerance': 1e-6,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update( params )

solver.solve()

z_output, omega_output, mu_output, rho_output, tau_output = psi.split( deepcopy=True )

xdmffile_z.write( z_output, 0 )
xdmffile_omega.write( omega_output, 0 )
xdmffile_mu.write( mu_output, 0 )
xdmffile_rho.write( rho_output, 0 )
xdmffile_tau.write( tau_output, 0 )

io.print_scalar_to_csvfile( z_output, (args.output_directory) + '/z.csv' )
io.print_vector_to_csvfile( omega_output, (args.output_directory) + '/omega.csv' )
io.print_scalar_to_csvfile( mu_output, (args.output_directory) + '/mu.csv' )
io.print_vector_to_csvfile( rho_output, (args.output_directory) + '/rho.csv' )
io.print_vector_to_csvfile( tau_output, (args.output_directory) + '/tau.csv' )

print( "BCs check: " )
print( f"\t<<(z - z_exact)^2>>_partial Omega = {termcolor.colored( msh.difference_on_boundary( z_output, z_exact ), 'red' )}" )
print(
    f"\t<<|omega - omega_exact|^2>>_partial Omega = {termcolor.colored( np.sqrt( assemble( (n[i] * omega_output[i] - n[i] * omega_exact[i]) ** 2 * ds ) / assemble( Constant( 1 ) * ds ) ), 'red' )}" )
print( f"\t<<(mu - mu_exact)^2>>_partial Omega = {termcolor.colored( msh.difference_on_boundary( mu_output, mu_exact ), 'red' )}" )
print(
    f"\t<<|rho - rho_exact|^2>>_partial Omega = {termcolor.colored( np.sqrt( assemble( (rho_output[i] - rho_exact[i]) * (rho_output[i] - rho_exact[i]) * ds ) / assemble( Constant( 1 ) * ds ) ), 'red' )}" )
print( f"\t<<(tau - tau_exact)^2>>_partial Omega = {termcolor.colored( msh.difference_on_boundary( tau_output, f ), 'red' )}" )

print( "Check that the PDE is satisfied: " )
print( f"\t<<(Nabla^2 partial_i ( z partial_i z) - f)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( tau_output, tau_exact ), 'green' )}" )

print( "Comparison with exact solution: " )
print( f"\t<<(z - z_exact)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( z_output, z_exact ), 'blue' )}" )
print(
    f"\t<<|omega - omega_exact|^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( project( sqrt( (omega_output[i] - omega_exact[i]) * (omega_output[i] - omega_exact[i]) ), Q_z ), project( Constant( 0 ), Q_z ) ), 'blue' )}" )
print( f"\t<<(mu - mu_exact)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( mu_output, mu_exact ), 'blue' )}" )
print(
    f"\t<<|rho - rho_exact|^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( project( sqrt( (rho_output[i] - rho_exact[i]) * (rho_output[i] - rho_exact[i]) ), Q_z ), project( Constant( 0 ), Q_z ) ), 'blue' )}" )
print( f"\t<<(tau - tau_exact)^2>>_Omega = {termcolor.colored( msh.difference_in_bulk( tau_output, tau_exact ), 'blue' )}" )

xdmffile_check.write( project( mu_output - mu_exact, Q_z ), 0 )
xdmffile_check.write( project( sqrt( (rho_output[i] - rho_exact[i]) * (rho_output[i] - rho_exact[i]) ), Q_z ), 0 )
xdmffile_check.write( project( tau_output - f, Q_z ), 0 )
xdmffile_check.close()
#
# msh.bulk_points( mesh )
