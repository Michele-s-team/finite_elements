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


#r, R must be the same as in generate_mesh.py
# R = 1.0
L = 2.2
h = L
r = 0.15
#these must be the same c_R, c_r as in generate_mesh.py, with the third component dropped
# c_R = [0.0, 0.0]
c_r = [L/2, h/2]



#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)

n  = FacetNormal(mesh)


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

# Define function spaces
#the '2' in ''P', 2)' is the order of the polynomials used to describe these spaces: if they are low, then derivatives high enough of the functions projected on thee spaces will be set to zero !
O = VectorFunctionSpace(mesh, 'P', 2, dim=2)
O3d = VectorFunctionSpace(mesh, 'P', 2, dim=3)
Q2 = FunctionSpace(mesh, 'P', 2)
#I will use Q4 for functions which involve high order derivatives
Q4 = FunctionSpace(mesh, 'P', 4)


# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 2.2)'
#the obstacle
cylinder = 'on_boundary && ((x[0]-1.1)*(x[0]-1.1) + (x[1]-1.1)*(x[1]-1.1) < (0.2*0.2))'


#  norm of vector x
def norm(x):
    return (np.sqrt(np.dot(x, x)))

#norm for UFL vectors
def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))

epsilon = ufl.PermutationSymbol(2)



#trial analytical expression for a vector
class TangentVelocityExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

#trial analytical expression for the height function z(x,y)
class ManifoldExpression(UserExpression):
    def eval(self, values, x):
        # values[0] = 4*x[0]*x[1]*sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
        # values[0] = sin(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
        #tentative smooth surface
        #         values[0] = ((norm(np.subtract(x, c_r)) - r)**4)  * ((norm(np.subtract(x, c_R)) - R)**4)
        # values[0] = 0.1 * x[0]*(8.0 - (6.0 * x[0])/L + (x[0]**3)/(L**3))
        values[0] = 10**(-3) * ((L - x[0])**3) * (x[0])**3
    def value_shape(self):
        return (1,)

# trial analytical expression for the  surface tension sigma(x,y)
class SurfaceTensionExpression(UserExpression):
        def eval(self, values, x):
            # values[0] = 4*x[0]*x[1]*sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
            # values[0] = cos(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
            values[0] = 0.0

        def value_shape(self):
            return (1,)

#trial analytical expression for w
class NormalVelocityExpression(UserExpression):
    def eval(self, values, x):
        # values[0] = (np.subtract(x, c_r)[0])*(np.subtract(x, c_r)[1])*cos(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
        values[0] = 0.0
    def value_shape(self):
        return (1,)


#analytical expression for a general scalar function
class ScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = cos(norm(np.subtract(x, c_r)) - r) * cos(norm(np.subtract(x, c_R)) - R)
    def value_shape(self):
        return (1,)
t=0

# Define trial and test functions
#v[i] = v^i_{notes} (tangential velocity)
v = TrialFunction(O)
#nu is the test function related to nu
nu = TestFunction(O)
#w = w_notes (normal velocity)
w = TrialFunction(Q2)
#o = omega_{notes} is the test function related to w
omega = TestFunction(Q2)
#sigma = \sigma_{notes}
sigma = TrialFunction(Q2)
#q  = q_{notes} is the test function related to sigma
q = TestFunction(Q2)
#z = z_notes
z = TrialFunction(Q4)
zeta = TestFunction(Q4)




#definition of scalar, vectorial and tensorial quantities
#latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)

def X(z):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([x[0], x[1], z])

#e(z)[i] = e_i_{al-izzi2020shear}
def e(z):
    return as_tensor([[1, 0, z.dx(0)], [0, 1, z.dx(1)]])

#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS
#normal(z) = \hat{n}_{al-izzi2020shear}
def normal(z):
    return as_tensor(cross(e(z)[0], e(z)[1]) /  ufl_norm(cross(e(z)[0], e(z)[1])) )
#MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS

#the normal to the two-dimensional manifold on the outflow
def n_outflow():
    return as_tensor([1.0, 0.0])


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

# def g(z):
#     return as_tensor(dot((e(z))[i], (e(z))[j]), (i, j))
def detg(z):
    return ufl.det(g(z))

def abs_detg(z):
    return np.abs(ufl.det(g(z)))

def sqrt_detg(z):
    return sqrt(detg(z))

def sqrt_abs_detg(z):
    return sqrt(abs_detg(z))

def sqrt_deth(z):
    return 1

def my_n(z):
    return as_tensor([1.0, 1.0])

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