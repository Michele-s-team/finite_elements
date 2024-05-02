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
parser.add_argument("T")
parser.add_argument("N")
args = parser.parse_args()

#CHANGE PARAMETERS HERE
#r, R must be the same as in generate_mesh.py
# R = 1.0
tol = 1E-3
L = 2.2
h = 10.0
r = 0.05
# R = 1.0
c_r = [0.2, h/2]
# c_r = [0, 0]
#CHANGE PARAMETERS HERE

#  norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))


#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mesh_coordinates = mesh.coordinates()

#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_rectangle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)

# ds = ds(metadata={'quadrature_degree': 2})

# n  = FacetNormal(mesh)

# print("Mesh points:")
# for x in mesh.coordinates():
#     print('\t%s' % x)


# example of how to move the mesh:
# for x in mesh.coordinates():
#     x[0] *= 2.0
# print("Mesh points after scaling:")
# for x in mesh.coordinates():
#     print('\t%s' % x)


# this class has the method `on` that tells whether the coordiante x lies on the inner circle of the mesh
class MyCircle(SubDomain):
    c = [], r

    def __init__(self, c_in, r_in):
        self.c = c_in
        self.r = r_in
        print("Constructed instance of MyCircle with c = ", (self.c), ", r = ", (self.r))

    def on(self, x):
        # here x is intended to be an ordinary array of floats
        if (abs(my_norm(x - (self.c)) - (self.r)) / (self.r)) < tol:
            return True
        else:
            return False


circle_r = MyCircle(c_r, r)
# circle_R = MyCircle(c_R, R)

# Print all vertices that belong to the boundary parts
# print("Mesh points on circle_r: ")
# for x in mesh.coordinates():
#     if circle_r.on(x): print('\t%s' % x)


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
Q = FunctionSpace(mesh, 'P', 1)
#I will use Q4 for functions which involve high order derivatives
Q4 = FunctionSpace(mesh, 'P', 4)

dofmap = Q4.dofmap()
nvertices = mesh.ufl_cell().num_vertices()

# # Set up a vertex_2_dof list
# indices = [dofmap.tabulate_entity_dofs(0, i)[0] for i in range(nvertices)]
# vertex_2_dof = dict()
# [vertex_2_dof.update(dict(vd for vd in zip(cell.entities(0),
#                                         dofmap.cell_dofs(cell.index())[indices])))
#                         for cell in cells(mesh)]



# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 10.0)'
cylinder = 'on_boundary && x[0]>0.0 && x[0]<0.4 && x[1]>4.0 && x[1]<6.0'
# inflow   = 'on_boundary && (x[0] < 0.0 + 0.001)'
# outflow  = 'on_boundary && (x[0] > 0.0 + 0.001)'
# cylinder = 'on_boundary && ((x[0]-0.0)*(x[0]-0.0) + (x[1]-0.0)*(x[1]-0.0) < (0.2*0.2))'
#CHANGE PARAMETERS HERE




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
        # values[0] = 0
        values[0] =  1E-3 * (x[0]*(x[0]-L)/L**2) *  (x[1]*(x[1]-h)/h**2)
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
        values[0] = cos(my_norm(np.subtract(x, c_r)) - r) * cos(my_norm(np.subtract(x, c_R)) - R)
    def value_shape(self):
        return (1,)
t=0

# Define trial and test functions
#v[i] = v^i_{notes} (tangential velocity)
v = TrialFunction(O)
#nu is the test function related to nu
nu = TestFunction(O)
#w = w_notes (normal velocity)
w = TrialFunction(Q)
#o = omega_{notes} is the test function related to w
omega = TestFunction(Q)
#sigma = \sigma_{notes}
sigma = TrialFunction(Q)
#q  = q_{notes} is the test function related to sigma
q = TestFunction(Q)
#z = z_notes
z = TrialFunction(Q4)
zeta = TestFunction(Q4)




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

#normal vector to the manifold pointing outwards the manifold. This vector field is defined everywhere in the manifold, but it makes sense only at the edges (and it should be used only at the edges)
#def n(z):
#    u = calc_normal_cg2(mesh)
#    # x = ufl.SpatialCoordinate(mesh)
#    # normalization = dot(u, u)
#    # zeta = g(z)[i,j]*u[i]*u[j]
#    c = 1.0/sqrt(g(z)[i,j]*u[i]*u[j])
#    return as_tensor(c*u[i], (i))


#normal vector to Omega  at the boundary between Omega and a boundary surface with tangent vector t. This is a proper vector in T_p(Omega) and it is normalized to unity accordng to the metric g
def n(z):
    u = calc_normal_cg2(mesh)
    hat_z = as_tensor([0, 0, 1])
    hat_n = as_tensor([u[0], u[1], 0])
    
    t = as_tensor(cross(hat_n, hat_z))
    c = as_tensor([-(e(z))[1, i]*t[i], (e(z))[0, i]*t[i]])
    return as_tensor(c[j]/sqrt(g_c(z)[k,l]*c[k]*c[l]), (j))

# def n_e(z):
#
#     x = ufl.SpatialCoordinate(mesh)
#     output = conditional(gt(x[0], 0.5), 1, 0) *  conditional(gt(x[1], 0.5), 1, 0)
#     return as_tensor([output, 0])


#the normal vector on the inflow and outflow
def n_inout(z):
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([conditional(lt(x[0], L/2), -1.0, 1.0), 0.0] )
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
