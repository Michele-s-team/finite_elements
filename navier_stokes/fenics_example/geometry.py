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
#r, R must be the same as in generate_mesh.py
# R = 1.0
tol = 1E-3
L = 1.0
h = 1.0
sigma0 = 10.0
# r = 0.05
# c_r = [0.2, 0.2]
#CHANGE PARAMETERS HERE

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)


# Define function spaces
P_z = FiniteElement('P', triangle, 1)
P_omega = VectorElement('P', triangle, 2)
element = MixedElement([P_z, P_omega])
Q_z_omega = FunctionSpace(mesh, element)
Q_z = Q_z_omega.sub(0).collapse()
Q_omega = Q_z_omega.sub(1).collapse()

#analytical expression for a general scalar function
class ScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        c_r = [0.2, 0.2]
        r = 0.05
        values[0] = cos(my_norm(np.subtract(x, c_r)) - r)**2.0 * np.sin((x[1]+1)/L) * np.cos((x[0]/h)**2)
    def value_shape(self):
        return (1,)


#  norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))




#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)


def calc_normal_cg2(mesh):
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * ds
    l = inner(-n, v) * ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)
    solve(A, nh.vector(), L)
    return nh






# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
in_flow   = 'near(x[0], 0)'
out_flow  = 'near(x[0], 1.0)'
in_out_flow    = 'near(x[0], 0) || near(x[0], 1.0)'
top_wall    = 'near(x[1], 1.0)'
bottom_wall   = 'near(x[1], 0)'
top_bottom_wall    = 'near(x[1], 0) || near(x[1], 1.0)'
boundary = 'on_boundary'
#CHANGE PARAMETERS HERE



#norm for UFL vectors
def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))

epsilon = ufl.PermutationSymbol(2)



#trial analytical expression for the height function z(x,y)
class z_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]/L * x[1]/h
    def value_shape(self):
        return (1,)


#trial analytical expression for a vector
class omega_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)
    
class sigma_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = sigma0 * x[1]*(h-x[1])/(h**2)
    def value_shape(self):
        return (1,)


        
        
t=0





#definition of scalar, vectorial and tensorial quantities
#latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)

def my_function():
    x = ufl.SpatialCoordinate(mesh)
    return(((x[0]-c_r[0])/my_norm(np.subtract(x, c_r)))**2)

def X(z):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([x[0], x[1], z])

#e(z)[i] = e_i_{al-izzi2020shear}
def e(z):
    return as_tensor([[1, 0, z.dx(0)], [0, 1, z.dx(1)]])

def e_p(z, x):
    return ([(project(e(z)[0], O3d))(x), (project(e(z)[1], O3d))(x)])

def normal_p(z, x):
    return ((project(normal(z), O3d))(x))

#radial vector for polar coordinates with origin at c_r (intended as a vector in R^3)
def hat_r():
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor(np.divide([x[0]-c_r[0], x[1]-c_r[1], 0.0], my_norm(np.subtract(x, c_r))))
    
#tangent vector for polar coordinates with origin at c_r (intended as a vector in R^3)
def hat_t():
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor(np.divide([-(x[1]-c_r[1]), x[0]-c_r[0], 0.0], my_norm(np.subtract(x, c_r))))


#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS
#normal(z) = \hat{n}_{al-izzi2020shear}
def normal(z):
    return as_tensor(cross(e(z)[0], e(z)[1]) /  ufl_norm(cross(e(z)[0], e(z)[1])) )
#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS




#b(z)[i,j] = b_{ij}_{al-izzi2020shear}
def b(z):
    return as_tensor((normal(z))[k] * (e(z)[i, k]).dx(j), (i,j))


#the gradient of z(x,y)
def grad_z(z):
    return as_vector(z.dx(i), (i))

#an example of a vector field obatined by contracting indexes
def my_vector_field(z):
    return as_vector(grad_z(z)[j]*g(z)[i,k]*g(z)[k, j] * dot(e(z)[0], normal(z)), (i))

#g_{ij}
def g(z):
    return as_tensor([[1+ (z.dx(0))**2, (z.dx(0))*(z.dx(1))],[(z.dx(0))*(z.dx(1)), 1+ (z.dx(1))**2]])

#g^{ij}
def g_c(z):
    return ufl.inv(g(z))

def detg(z):
    return ufl.det(g(z))

def abs_detg(z):
    return np.abs(ufl.det(g(z)))

def sqrt_detg(z):
    return sqrt(detg(z))

def sqrt_abs_detg(z):
    return sqrt(abs_detg(z))

def sqrt_deth(z):
    x = ufl.SpatialCoordinate(mesh)
    # #v = {\partial y^1/\partial x^\mu, \partial y^2/\partial x^\mu}_notesreall2013general
    # v_r = as_tensor([-(x[1]-c_r[1]), (x[0]-c_r[0])])
    # v_R = as_tensor([-(x[1]-c_R[1]), (x[0]-c_R[0])])
    #
    #
    c = conditional(lt(abs(x[0] - 0.0), tol), g(z)[1,1], 1.0) * \
        conditional(lt(abs(x[0] - L), tol), g(z)[1, 1], 1.0) * \
        conditional(lt(abs(x[1] - 0.0), tol), g(z)[0, 0], 1.0) * \
        conditional(lt(abs(x[1] - h), tol), g(z)[0, 0], 1.0)
    return sqrt(c)
    # return sqrt(c)
    # return 1


#normal vector to Omega  at the boundary between Omega and a boundary surface with tangent vector t. This is a proper vector in T_p(Omega) and it is normalized to unity accordng to the metric g
def n(z):
    u = calc_normal_cg2(mesh)
    hat_z = as_tensor([0, 0, 1])
    hat_n = as_tensor([u[0], u[1], 0])
    
    t = as_tensor(cross(hat_n, hat_z))
    c = as_tensor([-(e(z))[1, i]*t[i], (e(z))[0, i]*t[i]])
    return as_tensor(c[j]/sqrt(g_c(z)[k,l]*c[k]*c[l]), (j))


#the normal vector on the inflow and outflow normalized according to g and pointing outside Omega
def n_in_out(z):
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([conditional(lt(x[0], L/2), -1.0, 1.0), 0.0] )
    return as_tensor(u[k]/sqrt(g(z)[i,j]*u[i]*u[j]), (k))

#the normal vector on the top  and bottom wall normalized according to g and pointing outside Omega
def n_top_bottom(z):
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([0.0, conditional(lt(x[1], h/2), -1.0, 1.0)] )
    return as_tensor(u[k]/sqrt(g(z)[i,j]*u[i]*u[j]), (k))

#a normal vector pointing outwards the mesh
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
    return project(nh, O)


#H(z) = H_{al-izzi2020shear}
def H(z):
    return (0.5 * g_c(z)[i, j]*b(z)[j, i])

#K(z) = K_{al-izzi2020shear}
def K(z):
    return(ufl.det(as_tensor(b(z)[i,k]*g_c(z)[k,j], (i, j))))


#christoffel symbols: Gamma(z)[i,j,k] = {\Gamma^i_{jk}}_{al-izzi2020shear}
def Gamma(z):
    return as_tensor(0.5 * g_c(z)[i,l] * ( (g(z)[l, k]).dx(j) + (g(z)[j, l]).dx(k) - (g(z)[j, k]).dx(l) ), (i, j, k))

#covariant derivative of vector v with respect to \partial/partial x: Nabla_v(v, z)[i, j] = {\Nabla_j v^i}_{al-izzi2020shear}
def Nabla_v(u, z):
    return as_tensor((u[i]).dx(j) + u[k] * Gamma(z)[i, k, j], (i, j))

#covariant derivative of one-form f with respect to \partial/partial x: Nabla_f(f, z)[i, j] = {\Nabla_j f_i}_{al-izzi2020shear}
def Nabla_f(f, z):
    return as_tensor((f[i]).dx(j) - f[k] * Gamma(z)[k, i, j], (i, j))


#Laplace beltrami operator on a scalar function f: Nabla_LB(f) = {\Nabla_{LB} f}_{al-izzi2020shear}
def Nabla_LB(f, z):
    return (-g_c(z)[k,j] * Nabla_f(as_tensor(f.dx(i), (i)), z)[k, j])

#second definition of Laplace beltrami operator, equivalent to Nabla_LB
# def Nabla_LB2(f, z):
#     return (-1.0/sqrt_abs_detg(z) * (sqrt_abs_detg(z)*g_c(z)[i, j]*(f.dx(j))).dx(i))

def Nabla_LB_omega(omega, z):
    return as_tensor(- sqrt_abs_detg(z) * g_c(z)[j,k] * epsilon[j,i] * ((sqrt_abs_detg(z) * g_c(z)[l,m] * g_c(z)[n,omega] * epsilon[l,n] * ((omega[omega]).dx(m))).dx(k)), (i))


# Define symmetric gradient
def epsilon(u):
    # nabla_grad(u)_{i,j} = (u[j]).dx[i]
    #sym(nabla_grad(u)) =  nabla_grad(u)_{i,j} + nabla_grad(u)_{j,i}
    # return sym(nabla_grad(u))
    return as_tensor(0.5*(u[i].dx(j) + u[j].dx(i)), (i,j))

# Define stress tensor
def tensor_sigma(u, p):
    return as_tensor(2*epsilon(u)[i,j] - p*Identity(len(u))[i,j], (i, j))


#rate-of_deformation tensor: d(u, un, z)[i, j] = {d_{ij}}_{alizzi2020shear}
def d(u, un, z):
    return as_tensor(0.5 * ( g(z)[i, k]*Nabla_v(u, z)[k, j] + g(z)[j, k]*Nabla_v(u, z)[k, i] ) - (b(z)[i,j]) * un, (i, j))

#2-contravariant rate-of_deformation tensor: d_c(u, un, z)[i, j] = {d^{ij}}_{alizzi2020shear}
def d_c(u, un, z):
    return as_tensor(g_c(z)[i, k] * g_c(z)[j, l] * d(u, un, z)[k,l], (i,j))

#return the arithmetic mean between vectors a and b
def mean_v(a, b):
    return as_tensor(0.5 * (a[i]+b[i]), (i))


#the varaiation of the manifold height z over dt
def dzdt(v, w, z):
    return( v[i]*(e(z))[i, 2] + w*(normal(z))[2]   - (z.dx(j))*(v[i]*(e(z))[i, j] + w*(normal(z))[j])   )
