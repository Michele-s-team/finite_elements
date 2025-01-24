'''

run with
python3 solve.py [path where to read the mesh]
wxample:
clear; clear; rm -rf solution; python3 solve.py /home/fenics/shared/example-nitsches-method/2d/mesh
'''


from __future__ import print_function
from fenics import *
import argparse
import ufl as ufl
import sys

#add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import mesh as msh
import input_output as io



parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
args = parser.parse_args()

xdmffile_u = XDMFFile("solution/u.xdmf")

i, j, k, l = ufl.indices(4)


#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")

boundary = 'on_boundary'

Q = FunctionSpace( mesh, 'P', 2 )
V = VectorFunctionSpace(mesh, 'P', 2)

n = FacetNormal(mesh)

class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0 + (x[0])**2 + 2.0*((x[1])**2)
        values[1] = 1.0 + (x[0])**2 - 2.0*((x[1])**2)
    def value_shape(self):
        return (2,)

class grad_u0_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 2.0*x[0]
        values[1] = 4.0*x[1]
    def value_shape(self):
        return (4,)
    
class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        values[0] = 6.0
        values[1] = -2.0
    def value_shape(self):
        return (2,)
 
# Define variational problem
u = Function(V)
v = TestFunction(V)
f = Function(V)
grad_u_0 = Function(V)
grad_u_1 = Function(V)
g = Function(V)

eta = 10.0
f.interpolate(laplacian_u_expression(element=V.ufl_element()))
g.interpolate(u_expression(element=V.ufl_element()))
grad_u_0.interpolate(grad_u0_expression(element=V.ufl_element()))

F_0 = (((u[i]).dx(j))*((v[i]).dx(j)) + f[i]*v[i]) * dx - ( (n[i]*grad_u_0[i]) * v[0] + n[i]*((u[1]).dx(i))*v[1] ) * ds
F_N = ( eta * (n[j]*u[j] - n[j]*g[j])*(v[i]*n[i]) ) * ds
F = F_0 + F_N

solve(F == 0, u)

xdmffile_u.write(u, 0)
io.print_vector_to_csvfile(u, 'solution/u.csv')
