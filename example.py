'''
This code solves the Poisson equation Nabla u = f
run with

clear; clear; python3 example.py [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
example:
clear; clear; rm -rf solution; python3 example.py /home/fenics/shared/mesh /home/fenics/shared/solution
'''

from __future__ import print_function
from fenics import *
import argparse
from mshr import *
import numpy as np
import ufl as ufl

L=2.2
h=0.41

i, j = ufl.indices(2)

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

xdmffile_u = XDMFFile((args.output_directory) + "/u.xdmf")
xdmffile_error = XDMFFile((args.output_directory) + "/error.xdmf")

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")

# Define boundaries and obstacle
# CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_lr = 'near(x[0], 0) || near(x[0], 2.2)'
boundary_tb = 'near(x[1], 0) || near(x[1], 0.41)'
# CHANGE PARAMETERS HERE

#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)

#  norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))

#test for surface elements
ds_l = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)

#a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals( UserExpression ):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos(my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print("Integral l = ", assemble(f_test_ds*ds_l), " exact value = 0.373168")
print("Integral r = ", assemble(f_test_ds*ds_r), " exact value = 0.00227491")
print("Integral t = ", assemble(f_test_ds*ds_t), " exact value = 1.36138")
print("Integral b = ", assemble(f_test_ds*ds_b), " exact value = 1.02837")




n = FacetNormal(mesh)

Q = FunctionSpace( mesh, 'P', 8 )
V = VectorFunctionSpace( mesh, 'P', 8, dim=2 )

class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
    def value_shape(self):
        return (1,)

class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 2.0*x[0]
        # values[1] = 4.0*x[1]
        values[0] =  2 *(np.pi) *cos(2 *(np.pi) *((x[0]) - (x[1]))**2) * cos(2 *(np.pi) *((x[0]) + (x[1]))) + 4 *(np.pi) *(-(x[0]) + (x[1]))* sin(2 *(np.pi) * ((x[0]) - (x[1]))**2) * sin(2 * (np.pi) * ((x[0]) + (x[1])))
        values[1] = 2 * (np.pi) * cos(2* (np.pi) * ((x[0]) - (x[1]))**2) * cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4* (np.pi) * ((x[0]) - (x[1])) * sin(2 *(np.pi) *((x[0]) - (x[1]))**2) * sin(2 * (np.pi)*  ((x[0]) + (x[1])))
    def value_shape(self):
        return (2,)
    
class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 6.0
        values[0] = 8 *(np.pi)* (-(np.pi)* (1+4* (x[0]-(x[1]))**2) * cos(2* (np.pi)* (x[0]-(x[1]))**2)-sin(2* (np.pi) *(x[0]-(x[1]))**2))* sin(2* (np.pi)* (x[0]+(x[1])))
    def value_shape(self):
        return (1,)


# Define variational problem
u = Function( Q )
nu = TestFunction( Q )
f = Function( Q )
grad_u = Function( V )
J_u = TrialFunction( Q )

u.interpolate( u_expression( element=Q.ufl_element() ) )
grad_u.interpolate( grad_u_expression( element=V.ufl_element() ) )
f.interpolate( laplacian_u_expression( element=Q.ufl_element() ) )

u_profile = Expression( 'sin(2.0*pi*(x[0]+x[1])) * cos(2.0*pi*pow(x[0]-x[1], 2))', L=L, h=h, element=Q.ufl_element() )
bc_u = DirichletBC( Q, u_profile, boundary_tb )


F = (dot( grad(u), grad( nu ) ) + f * nu) * dx - dot( n, grad_u ) * nu * (ds_l + ds_r) - n[i]*(u.dx(i)) * nu * (ds_t + ds_b)
bcs= [bc_u]
J = derivative( F, u, J_u )
problem = NonlinearVariationalProblem( F, u, bcs, J )
solver = NonlinearVariationalSolver( problem )


#set the solver parameters here
params = {'nonlinear_solver': 'newton',
           'newton_solver':
            {
                'linear_solver'           : 'mumps',
                'absolute_tolerance'      : 1e-6,
                'relative_tolerance'      : 1e-6,
                'maximum_iterations'      : 1000000,
                'relaxation_parameter'    : 0.95,
             }
}
solver.parameters.update(params)

solver.solve()

xdmffile_u.write( u, 0 )
xdmffile_error.write( project(u.dx(i).dx(i) - f, Q) , 0)

print("\int (n[i] \partial_i u - n[i] grad_u[i])^2 dS = ", assemble( ((n[i]*grad_u[i]) - (n[i] * u.dx( i ))) ** 2 * (ds_l + ds_r) ) )