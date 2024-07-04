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

V = FunctionSpace(mesh, 'P', 1)
O = VectorFunctionSpace(mesh, 'P', 2, dim=2)

class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 2.0*x[0]
        # values[1] = 4.0*x[1]
        values[0] = 2.0*x[0]
        values[1] = 6.0*((x[1])**2)
    def value_shape(self):
        return (2,)
    
class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 6.0
        values[0] = 2.0 + 12.0 * x[1]
        
    def value_shape(self):
        return (1,)



 
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
grad_u = Function(O)

grad_u = interpolate(grad_u_expression(element=O.ufl_element()), O)
f = interpolate(laplacian_u_expression(element=V.ufl_element()), V)




a = dot(grad(u), grad(v))*dx
L = -f*v*dx + dot(n,grad_u)*v*ds

u = Function(V)
solve(a == L, u)

xdmffile_u.write(u, 0)