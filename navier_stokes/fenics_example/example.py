"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u_D = 1 + x^2 + 2y^2
    f = -6
"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()


xdmffile_u = XDMFFile((args.output_directory) + "/u.xdmf")


# Create mesh and define function space
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


#trial analytical expression for a vector
class h_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 2.0*x[0]
        values[1] = 4.0*x[1]
    def value_shape(self):
        return (2,)


h = Function(O)
h = interpolate(h_expression(element=O.ufl_element()), O)


def boundary(x, on_boundary):
    return on_boundary
 
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx + dot(n,h)*v*ds

# Compute solution
u = Function(V)
solve(a == L, u)

xdmffile_u.write(u, 0)


