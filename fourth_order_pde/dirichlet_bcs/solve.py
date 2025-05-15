
'''
This code solves the biharmonic equation Nabla Nabla \partial_i (z \partial_i z) = f expressed in terms of the function
- z
- omega[i] = \partial_i z
- mu = \partial_i (z omega_i)
- rho_i = \partial_i mu
- tau = \partial_i rho_i

where the BCs for mu, rho and tau are imposed as Dirichlet BCs with respect to the exact solution, which is known in this case. 

run with
    python3 solve.py [problem name] [path where to read the mesh generated from generate_square_mesh.py or generate_ring_mesh.py] [path where to store the solution]
example:
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/fourth_order_pde/dirichlet_bcs/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_dirichlet $MESH_PATH $SOLUTION_PATH
    MESH_PATH="/home/fenics/shared/generate_mesh/2d/ring/solution"; SOLUTION_PATH="/home/fenics/shared/fourth_order_pde/dirichlet_bcs/solution"; rm -rf $SOLUTION_PATH; python3 solve.py ring_nitsche $MESH_PATH $SOLUTION_PATH
'''
import argparse
from fenics import *
import importlib
import termcolor
import numpy as np
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import function_spaces as fsp
import geometry as geo
import input_output as io
import mesh as msh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

#
# '''
# #square mesh
# # CHANGE PARAMETERS HERE
# L = 1.0
# h = 1.0
# # CHANGE PARAMETERS HERE
# '''
# #
# # ring mesh
# # CHANGE PARAMETERS HERE
# r = 1.0
# R = 2.0
# # CHANGE PARAMETERS HERE
# #


#
#
# parser = argparse.ArgumentParser()
# parser.add_argument( "input_directory" )
# parser.add_argument( "output_directory" )
# args = parser.parse_args()



# read the mesh
# mesh = Mesh()
# xdmf = XDMFFile( mesh.mpi_comm(), (args.input_directory) + "/triangle_mesh.xdmf" )
# xdmf.read( mesh )

# # radius of the smallest cell in the mesh
# r_mesh = mesh.hmin()

# print( f"Mesh radius = {r_mesh}" )

# # read the triangles
# mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() )
# with XDMFFile( (args.input_directory) + "/triangle_mesh.xdmf" ) as infile:
#     infile.read( mvc, "name_to_read" )
# cf = dolfin.cpp.mesh.MeshFunctionSizet( mesh, mvc )
# xdmf.close()
#
# # read the lines
# mvc = MeshValueCollection( "size_t", mesh, mesh.topology().dim() - 1 )
# with XDMFFile( (args.input_directory) + "/line_mesh.xdmf" ) as infile:
#     infile.read( mvc, "name_to_read" )
# sf = dolfin.cpp.mesh.MeshFunctionSizet( mesh, mvc )
# xdmf.close()
#
# # Define boundaries and obstacle
# # CHANGE PARAMETERS HERE
# boundary = 'on_boundary'
# # CHANGE PARAMETERS HERE


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

# # ring mesh
# #
# dx = Measure( "dx", domain=mesh, subdomain_data=cf, subdomain_id=1 )
# ds_r = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=2 )
# ds_R = Measure( "ds", domain=mesh, subdomain_data=sf, subdomain_id=3 )
# ds = ds_r + ds_R
# #

# a function space used solely to define f_test_ds
# Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
# f_test_ds = Function( Q_test )


# analytical expression for a  scalar function used to test the ds
# class FunctionTestIntegrals( UserExpression ):
#     def eval(self, values, x):
#         c_test = [0.3, 0.76]
#         r_test = 0.345
#         values[0] = cos( geo.my_norm( np.subtract( x, c_test ) ) - r_test ) ** 2.0
#
#     def value_shape(self):
#         return (1,)


# f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )

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
# msh.test_mesh_integral( 2.90212, f_test_ds, dx, '\int f dx' )
# msh.test_mesh_integral( 2.77595, f_test_ds, ds_r, '\int_r f ds' )
# msh.test_mesh_integral( 3.67175, f_test_ds, ds_R, '\int_R f ds' )
#

# n = FacetNormal( mesh )






J = derivative( vp.F, fsp.psi, fsp.J_Q )
problem = NonlinearVariationalProblem( vp.F, fsp.psi, vp.bcs, J )
solver = NonlinearVariationalSolver( problem )
# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  # 'linear_solver': 'superlu',
                  'linear_solver': 'mumps',
                  'absolute_tolerance': 1e-12,
                  'relative_tolerance': 1e-12,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update( params )

solver.solve()

prout_bc = importlib.import_module(swi.prout_bc)

