'''

run with

clear; clear; python3 example.py [path where to read the mesh generated from generate_mesh.py] [path where to store the solution]
example:
clear; clear; rm -rf solution; python3 example.py /home/fenics/shared/mesh /home/fenics/shared/solution
'''

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import argparse
import numpy as np
import ufl as ufl

i, j = ufl.indices(2)

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

xdmffile_u = XDMFFile((args.output_directory) + "/u.xdmf")

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")

n = FacetNormal(mesh)

Q = FunctionSpace( mesh, 'P', 8 )
V = VectorFunctionSpace( mesh, 'P', 8, dim=2 )

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

grad_u.interpolate( grad_u_expression( element=V.ufl_element() ) )
f.interpolate( laplacian_u_expression( element=Q.ufl_element() ) )

F = (dot( grad(u), grad( nu ) ) + f * nu) * dx - dot( n, grad_u ) * nu * ds
bcs= []
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


# solve( a == L, u )

xdmffile_u.write( u, 0 )

print("\int (n[i] \partial_i u - n[i] grad_u[i])^2 dS = ", assemble( ((n[i]*grad_u[i]) - (n[i] * u.dx( i ))) ** 2 * ds ) )