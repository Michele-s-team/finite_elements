'''

run with

clear; clear; python3 example.py [path where to read the mesh generated from generate_mesh.py]
example:
clear; clear; python3 example.py /home/fenics/shared/mesh/membrane_mesh
Solving linear variational problem.

'''

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

xdmffile_u = XDMFFile("u.xdmf")

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
u_ = Function( Q )
u = TrialFunction( Q )
v = TestFunction( Q )
f = Function( Q )
grad_u = Function( V )

grad_u.interpolate( grad_u_expression( element=V.ufl_element() ) )
f.interpolate( laplacian_u_expression( element=Q.ufl_element() ) )

a = dot(grad(u), grad(v))*dx
L = -f*v*dx + dot(n,grad_u)*v*ds

solve(a == L, u_)

xdmffile_u.write(u_, 0)