import math

from fenics import *
from mshr import *
import numpy as np
# from dolfin import *
import meshio
import ufl as ufl
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

#CHANGE PARAMETERS HERE
r = 0.5
kappa = 1.0
eta = 1000.0
sigma0 = 1.0
C = 0.5
tol = 1E-3
#CHANGE PARAMETERS HERE



#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)

n_facet = FacetNormal(mesh)

# Define function spaces
P_z = FiniteElement('P', triangle, 1)
P_omega = VectorElement('P', triangle, 3)
element = MixedElement([P_z, P_omega])
Q_z_omega = FunctionSpace(mesh, element)
Q_z = Q_z_omega.sub(0).collapse()
Q_omega = Q_z_omega.sub(1).collapse()

#analytical expression for a general scalar function
class ScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        cs = [0.1, 0.5]
        rs = 1.0
        values[0] = cos(my_norm(np.subtract(x, cs)) - rs)**2.0 
    def value_shape(self):
        return (1,)


#  norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))


def C_n(i):
    return( C_min + (C_max-C_min)*i/(N-1))


#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)


# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_square = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) > 0.6'
boundary_circle = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) < 0.6'
#CHANGE PARAMETERS HERE



#norm for UFL vectors
def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))

epsilon = ufl.PermutationSymbol(2)


#CHANGE PARAMETERS HERE

#trial analytical expression for the height function z(x,y)
class z_Expression(UserExpression):
    def eval(self, values, x):
        # values[0] = C * cos(2*pi*x[0]/L) * ((x[1])**2)/2.0
        values[0] = 0
    def value_shape(self):
        return (1,)


#trial analytical expression for a vector
class grad_circle_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = C*x[0]
        values[1] = C*x[1]
        return (2,)
    
class grad_square_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0
    def value_shape(self):
        return (2,)
    
class sigma_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = sigma0
    def value_shape(self):
        return (1,)

#CHANGE PARAMETERS HERE


#definition of scalar, vectorial and tensorial quantities
#latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)


def X(z):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([x[0], x[1], z])

#e(z)[i] = e_i_{al-izzi2020shear}
def e(omega):
    return as_tensor([[1, 0, omega[0]], [0, 1, omega[1]]])


#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS
#normal(z) = \hat{n}_{al-izzi2020shear}
def normal(omega):
    return as_tensor(cross(e(omega)[0], e(omega)[1]) /  ufl_norm(cross(e(omega)[0], e(omega)[1])) )
#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS


#b(z)[i,j] = b_{ij}_{al-izzi2020shear}
def b(omega):
    return as_tensor((normal(omega))[k] * (e(omega)[i, k]).dx(j), (i,j))


#g_{ij}
def g(omega):
    return as_tensor([[1+ (omega[0])**2, (omega[0])*(omega[1])],[(omega[0])*(omega[1]), 1+ (omega[1])**2]])

#g^{ij}
def g_c(omega):
    return ufl.inv(g(omega))

def detg(omega):
    return ufl.det(g(omega))

def abs_detg(omega):
    return np.abs(ufl.det(g(omega)))

def sqrt_detg(omega):
    return sqrt(detg(omega))

def sqrt_abs_detg(omega):
    return sqrt(abs_detg(omega))


#the vector used to define the pull-back of the metric, h
def w():
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([-x[1], x[0]])

def sqrt_deth(omega):
    return(sqrt((w())[i]*(w())[j]*g(omega)[i, j]))

#the normal vector on the inflow and outflow normalized according to g and pointing outside Omega
def n_lr(omega):
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([conditional(lt(x[0], 0.0), -1.0, 1.0), 0.0] )
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))

#the normal vector on the top  and bottom wall normalized according to g and pointing outside Omega
def n_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([0.0, conditional(lt(x[1], 0.0), -1.0, 1.0)] )
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))


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


def n(omega):
    u = n_facet
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))

def n_smooth(omega):
    u = calc_normal_cg2(mesh)
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))


def n_facet_smooth():
    u = calc_normal_cg2(mesh)
    return as_tensor(u[k], (k))

#H(z) = H_{al-izzi2020shear}
def H(omega):
    return (0.5 * g_c(omega)[i, j]*b(omega)[j, i])

#K(z) = K_{al-izzi2020shear}
def K(omega):
    return(ufl.det(as_tensor(b(omega)[i,k]*g_c(omega)[k,j], (i, j))))


#christoffel symbols: Gamma(z)[i,j,k] = {\Gamma^i_{jk}}_{al-izzi2020shear}
def Gamma(omega):
    return as_tensor(0.5 * g_c(omega)[i,l] * ( (g(omega)[l, k]).dx(j) + (g(omega)[j, l]).dx(k) - (g(omega)[j, k]).dx(l) ), (i, j, k))

#covariant derivative of vector v with respect to \partial/partial x: Nabla_v(v, z)[i, j] = {\Nabla_j v^i}_{al-izzi2020shear}
def Nabla_v(u, omega):
    return as_tensor((u[i]).dx(j) + u[k] * Gamma(omega)[i, k, j], (i, j))

#covariant derivative of one-form f with respect to \partial/partial x: Nabla_f(f, z)[i, j] = {\Nabla_j f_i}_{al-izzi2020shear}
def Nabla_f(f, omega):
    return as_tensor((f[i]).dx(j) - f[k] * Gamma(omega)[k, i, j], (i, j))