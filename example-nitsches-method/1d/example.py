'''
This code solves the Poisson equation with Dirichlet BCs

u = u_D on \partial \Omega

by imposing the BCs with Nitsche's method. Run with  

run with
python3 example.py

'''

from __future__ import print_function
from fenics import *
import argparse
from mshr import *
import ufl as ufl

i, j, k, l = ufl.indices(4)

parser = argparse.ArgumentParser()
args = parser.parse_args()

xdmffile_u = XDMFFile("u.xdmf")


# Create mesh
channel = Rectangle(Point(0, 0), Point(1.0, 1.0))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
domain = channel
alpha = Constant(10.0)
mesh = generate_mesh(domain, 16)

h = CellDiameter(mesh)

V = FunctionSpace(mesh, 'P', 8)

#read mesh
# mesh=Mesh()
# with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
#     infile.read(mesh)
# mvc = MeshValueCollection("size_t", mesh, 2)
# with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
#     infile.read(mvc, "name_to_read")

def calc_normal_cg2(mesh):
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * ds
    l = inner(n, v) * ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)
    solve(A, nh.vector(), L)
    return nh

def my_n():
    u = calc_normal_cg2(mesh)
    return as_tensor(u[k], (k))

n = FacetNormal(mesh)


class u_expression(UserExpression):
    def eval(self, values, x):
        values[0] =cos(x[0])**2 + 2*sin(x[0]+x[1])**4
    def value_shape(self):
        return (1,)

# class grad_u_expression(UserExpression):
#     def eval(self, values, x):
#         # values[0] = 2.0*x[0]
#         # values[1] = 4.0*x[1]
#         values[0] =  2 *(np.pi) *cos(2 *(np.pi) *((x[0]) - (x[1]))**2) * cos(2 *(np.pi) *((x[0]) + (x[1]))) + 4 *(np.pi) *(-(x[0]) + (x[1]))* sin(2 *(np.pi) * ((x[0]) - (x[1]))**2) * sin(2 * (np.pi) * ((x[0]) + (x[1])))
#         values[1] = 2 * (np.pi) * cos(2* (np.pi) * ((x[0]) - (x[1]))**2) * cos(2 * (np.pi) * ((x[0]) + (x[1]))) + 4* (np.pi) * ((x[0]) - (x[1])) * sin(2 *(np.pi) *((x[0]) - (x[1]))**2) * sin(2 * (np.pi)*  ((x[0]) + (x[1])))
#     def value_shape(self):
#         return (2,)
    
class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # values[0] = 6.0
        values[0] = - 2 * (cos(2 * x[0]) - 4 * cos(2 * (x[0]+x[1])) + 4 * cos(4 * (x[0]+x[1])))
    def value_shape(self):
        return (1,)
 
# Define variational problem
u = Function(V)
u_D = Function(V)
v = TestFunction(V)
f = Function(V)

f.interpolate(laplacian_u_expression(element=V.ufl_element()))
u_D.interpolate(u_expression(element=V.ufl_element()))

#this is the ordinary variational functional
F_0 = dot(grad(u), grad(v))*dx + f*v*dx + ( - n[i]*(u.dx(i))  * v ) * ds
#this is the term that enforces the BCs with Nitche's method
F_N = ((+ alpha / h * (u - u_D)) * v - n[i] * (v.dx( i )) * (u - u_D)) * ds
F = F_0 + F_N

solve(F == 0, u)

print("\int_{\partial \Omnega} (u - u_D)^2 dS = ", assemble((u - u_D)**2*ds))

xdmffile_u.write(u, 0)