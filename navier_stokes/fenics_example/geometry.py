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
L = 2.2
h = 0.41
r = 0.05
c_r = [L/2.0, h/2.0]
# time step size
#bending rigidity
kappa = 1.0
#density
rho = 1.0
#viscosity
eta = 1.0
#Nitche's parameter
alpha = 1e3
tol = 1E-3
#CHANGE PARAMETERS HERE



#read mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")


#this is the facet normal vector, which cannot be plotted as a field
#n_overline = \overline{n}_notes_{on the circle}
n_overline = FacetNormal(mesh)

# Define function spaces
#finite elements for sigma .... omega
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_sigma_n = FiniteElement( 'P', triangle, 1 )
P_omega_n = VectorElement( 'P', triangle, 3 )
P_z_n = FiniteElement( 'P', triangle, 1 )

element = MixedElement( [P_v_n, P_w_n, P_sigma_n, P_omega_n, P_z_n] )
#total function space
Q = FunctionSpace(mesh, element)
#function spaces for vbar .... zn
Q_v = Q.sub( 0 ).collapse()
Q_w = Q.sub( 1 ).collapse()
Q_sigma = Q.sub( 2 ).collapse()
Q_omega = Q.sub( 3 ).collapse()
Q_z= Q.sub( 4 ).collapse()


#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos(my_norm(np.subtract(x, c_test)) - r_test)**2.0 
    def value_shape(self):
        return (1,)


# norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))


#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)


# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = 'near(x[0], 0.0)'
boundary_r  = 'near(x[0], 2.2)'
boundary_lr  = 'near(x[0], 0) || near(x[0], 2.2)'
boundary_tb  = 'near(x[1], 0) || near(x[1], 0.41)'
boundary_square = 'on_boundary && sqrt(pow(x[0] - 2.2/2.0, 2) + pow(x[1] - 0.41/2.0, 2)) > 0.1'
boundary_circle = 'on_boundary && sqrt(pow(x[0] - 2.2/2.0, 2) + pow(x[1] - 0.41/2.0, 2)) < 0.1'
#CHANGE PARAMETERS HERE

#norm for UFL vectors
def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))

epsilon = ufl.PermutationSymbol(2)

#CHANGE PARAMETERS HERE
class TangentVelocityExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

class NormalVelocityExpression(UserExpression):
        def eval(self, values, x):
            values[0] = 0.0
        def value_shape(self):
            return (1,)

class SurfaceTensionExpression(UserExpression):
        def eval(self, values, x):
            values[0] = 0.0
        def value_shape(self):
            return (1,)

class ManifoldExpression(UserExpression):
        def eval(self, values, x):
            values[0] = 0.0
        def value_shape(self):
            return (1,)

class OmegaExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

#profiles for the normal derivative
class grad_circle_Expression(UserExpression):
    def eval(self, values, x):
        a = 0.1
        values[0] = a * (x[0] - c_r[0])/my_norm(x-c_r)
        values[1] = a * (x[1] - c_r[1])/my_norm(x-c_r)
    def value_shape(self):
        return (2,)
    
class grad_square_Expression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0
    def value_shape(self):
        return (2,)
#CHANGE PARAMETERS HERE


#definition of scalar, vectorial and tensorial quantities
#latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)

#sign
#the vector of the differential manifold, which is equal to \vec{X}_{\Gamma}(x_1, x_2) on page 8 if al-izzi2020shear
def X(z):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([x[0], x[1], z])

#the vectors tangent to the curvilinear coordinates on the manifold : e(z)[i] = e_i_{al-izzi2020shear}
def e(omega):
    return as_tensor([[1, 0, omega[0]], [0, 1, omega[1]]])


#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS
#normal(z) = \hat{n}_{al-izzi2020shear}
def normal(omega):
    return as_tensor(cross(e(omega)[0], e(omega)[1]) /  ufl_norm(cross(e(omega)[0], e(omega)[1])) )
#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS

#first fundamental form: b(z)[i,j] = b_{ij}_{al-izzi2020shear}
def b(omega):
    return as_tensor((normal(omega))[k] * (e(omega)[i, k]).dx(j), (i,j))

#two-covariant metric tensor: g_{ij}
def g(omega):
    return as_tensor([[1+ (omega[0])**2, (omega[0])*(omega[1])],[(omega[0])*(omega[1]), 1+ (omega[1])**2]])

#two-contravariant metric tensor: g^{ij}
def g_c(omega):
    return ufl.inv(g(omega))

#determinant of the two-covariant metric tensor
def detg(omega):
    return ufl.det(g(omega))

#absolute value of the two-covariant metric tensor
def abs_detg(omega):
    return np.abs(ufl.det(g(omega)))

#square root of the determinant of the two-covariant metric tensor
def sqrt_detg(omega):
    return sqrt(detg(omega))

#square root of the absolute value of the two-covariant metric tensor
def sqrt_abs_detg(omega):
    return sqrt(abs_detg(omega))

#vector used to define the pull-back of the metric, h, on a circle with radius r centered at c ( it is independent of r), see 'notes reall2013general'
def dydtheta(c):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([-(x[1]-c[1]), x[0]-c[0]])

#square root of the determinant of the pull-back of the metric, h, on a circle with radius r centered at c ( it is independent of r)
def sqrt_deth_circle(omega, c):
    return(sqrt((dydtheta(c))[i]*(dydtheta(c))[j]*g(omega)[i, j]))

#square root of the determinant of the pull-back of the metric on a rectangular boundary given by a rectangle (0,0) - (L,0) - (L, h) - (0,h)
def sqrt_deth_square(omega):
    x = ufl.SpatialCoordinate(mesh)

    c = conditional(lt(abs(x[0] - 0.0), tol), g(omega)[1,1], 1.0) * \
        conditional(lt(abs(x[0] - L), tol), g(omega)[1, 1], 1.0) * \
        conditional(lt(abs(x[1] - 0.0), tol), g(omega)[0, 0], 1.0) * \
        conditional(lt(abs(x[1] - h), tol), g(omega)[0, 0], 1.0)
    return sqrt(c)

#the normal vector on the l and r boundaries of a rectangle  (0,0) - (L,0) - (L, h) - (0,h), normalized according to g and pointing outside Omega
def n_lr(omega):
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0] )
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))

#the normal vector on the t  and b boundaries of a rectangle  (0,0) - (L,0) - (L, h) - (0,h), normalized according to g and pointing outside Omega
def n_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0)] )
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))

#the facet normal vector on the l and r boundaries of a rectangle  (0,0) - (L,0) - (L, h) - (0,h)
#n_overline_lr() = \overline{n}_notes on the l and r edges of the rectangle
def n_overline_lr():
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0] )
    return as_tensor(u[k], (k))

#the facet normal vector on the t  and b boundaries of a rectangle  (0,0) - (L,0) - (L, h) - (0,h)
#n_overline_tb() = \overline{n}_notes on the t and b edges of the rectangle
def n_overline_tb():
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0)] )
    return as_tensor(u[k], (k))

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

#the normal to the manifold pointing outwards the manifold and normalized according to g, which cannot be plotted as a field
def n(omega):
    u = n_overline
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))

#the normal to the manifold pointing outwards the manifold and normalized according to g, which can be plotted as a field
def n_smooth(omega):
    u = calc_normal_cg2(mesh)
    return as_tensor(u[k]/sqrt(g(omega)[i,j]*u[i]*u[j]), (k))

#the normal to the manifold pointing outwards the manifold and normalized according to the Euclidean metric, which can be plotted as a field
def n_overline_smooth():
    u = calc_normal_cg2(mesh)
    return as_tensor(u[k], (k))

#mean curvature, H = H_{al-izzi2020shear}
def H(omega):
    return (0.5 * g_c(omega)[i, j]*b(omega)[j, i])

#gaussian curvature: K = K_{al-izzi2020shear}
def K(omega):
    return(ufl.det(as_tensor(b(omega)[i,k]*g_c(omega)[k,j], (i, j))))

#Christoffel symbols of the second kind related to g: Gamma(omega)[i,j,k] = {\Gamma^i_{jk}}_{al-izzi2020shear}
def Gamma(omega):
    return as_tensor(0.5 * g_c(omega)[i,l] * ( (g(omega)[l, k]).dx(j) + (g(omega)[j, l]).dx(k) - (g(omega)[j, k]).dx(l) ), (i, j, k))

#covariant derivative of vector v with respect to \partial/partial x and with respect to the Levi-Civita connection generated by g: Nabla_v(v, omega)[i, j] = {\Nabla_j v^i}_{al-izzi2020shear}
def Nabla_v(u, omega):
    return as_tensor((u[i]).dx(j) + u[k] * Gamma(omega)[i, k, j], (i, j))

#covariant derivative of one-form f with respect to \partial/partial x and with respect to the Levi-Civita connection generated by g: Nabla_f(f, omega)[i, j] = {\Nabla_j f_i}_{al-izzi2020shear}
def Nabla_f(f, omega):
    return as_tensor((f[i]).dx(j) - f[k] * Gamma(omega)[k, i, j], (i, j))

# #2-contravariant rate-of_deformation tensor: d_c(u, omega)[i, j] = {d^{ij}}_{alizzi2020shear for zero w}
# def d_c(v, omega):
#     return as_tensor( g_c(omega)[i, k] * g_c(omega)[j, l] * d( v, omega )[k, l], (i, j) )

#2-covariant rate-of_deformation tensor for zero normal velocity: d(u, z)[i, j] = {d_{ij}}_{alizzi2020shear for zero w}
def d(v, w, omega):
    return as_tensor( 0.5 * (g(omega)[i, k] * Nabla_v(v, omega)[k, j] + g(omega)[j, k] * Nabla_v(v, omega)[k, i]) - (b(omega)[i,j]) * w, (i, j) )

#2-contravariant rate-of_deformation tensor: d_c(u, un, z)[i, j] = {d^{ij}}_{alizzi2020shear}
def d_c(v, w, omega):
    return as_tensor( g_c(omega)[i, k] * g_c(omega)[j, l] * d(v, w, omega)[k,l], (i, j) )

#given a vector and a scalar, return the vector vector^i * scalar
def vector_times_scalar(vector, scalar):
    return as_tensor(scalar * vector[i], (i))
