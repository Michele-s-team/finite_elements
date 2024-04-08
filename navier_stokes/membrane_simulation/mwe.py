from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import meshio
import ufl as ufl
from geometry import *


mesh=Mesh()
with XDMFFile("triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")

V3d = VectorFunctionSpace(mesh, 'P', 2, dim=3)
Q = FunctionSpace(mesh, 'P', 2)

def norm(x):
    return (np.sqrt(np.dot(x, x)))
def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))

class MyScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]*x[1]*sin(4*(norm(np.subtract(x, c_r)) - r))*sin(4*(norm(np.subtract(x, c_R)) - R))
        # values[0] = sin(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
    def value_shape(self):
        return (1,)

i, j, k, l = ufl.indices(4)

def X(z):
    return as_tensor([x[0], x[1], z(x)])

def e(z):
    return as_tensor([[1, 0, z.dx(0)], [0, 1, z.dx(1)]])

#path for mac
output_directory = "/home/fenics/shared/navier_stokes/membrane_simulation/solution"

xdmffile_geometry = XDMFFile(output_directory + "/geometry.xdmf")


xdmffile_geometry = XDMFFile(output_directory + "/geometry.xdmf")
z_  = Function(Q)
z_ = interpolate(MyScalarFunctionExpression(element=Q.ufl_element()), Q)
xdmffile_geometry.write(project(z_, Q), 0)
xdmffile_geometry.write(project(X(z_), V3d), 0)

