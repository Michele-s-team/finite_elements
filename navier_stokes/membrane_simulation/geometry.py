from fenics import *
from mshr import *
import numpy as np
# from dolfin import *
import meshio
import ufl as ufl

#r, R must be the same as in generate_mesh.py
R = 1.0
r = 0.25
#these must be the same c_R, c_r as in generate_mesh.py, with the third component dropped
c_R = [0.0, 0.0]
c_r = [0.0, -0.1]

#paths for mac
input_directory = "/home/fenics/shared/mesh/membrane_mesh"
#paths for abacus
# input_directory = "/mnt/beegfs/home/mcastel1/navier_stokes"


#create mesh
mesh=Mesh()
with XDMFFile(input_directory + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(input_directory + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)

n  = FacetNormal(mesh)


# Define function spaces
#the '2' in ''P', 2)' is the order of the polynomials used to describe these spaces: if they are low, then derivatives high enough of the functions projected on thee spaces will be set to zero !
V = VectorFunctionSpace(mesh, 'P', 2)
V3d = VectorFunctionSpace(mesh, 'P', 2, dim=3)
Q = FunctionSpace(mesh, 'P', 2)


# Define boundaries
#a semi-circle given by the left half of circle_R
inflow   = 'on_boundary && (x[0] < 0.01) && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#a semi-circle given by the right half of circle_R
outflow   =  'on_boundary && (x[0] > 0.01) && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#the whole circle_R
external_boundary = 'on_boundary && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#the obstacle
cylinder = 'on_boundary && (x[0]*x[0] + x[1]*x[1] < (0.5*0.5))'


#  norm of vector x
def norm(x):
    return (np.sqrt(np.dot(x, x)))

#norm for UFL vectors
def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))


#analytical expression for a vector
class MyVectorFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = x[0]
        values[1] = -x[1]
    def value_shape(self):
        return (2,)
#analytical expression for a function
class MyScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 4*x[0]*x[1]*sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
        # values[0] = sin(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
    def value_shape(self):
        return (1,)
t=0

# Define trial and test functions
#u[i] = v^i_{notes} (tangential velocity)
u = TrialFunction(V)
v = TestFunction(V)
#w = w_notes (normal velocity)
w = TrialFunction(Q)
o = TestFunction(Q)
#p = \sigma_{notes}
p = TrialFunction(Q)
q = TestFunction(Q)
#z = z_notes
z = TrialFunction(Q)
x = TestFunction(Q)




#definition of scalar, vectorial and tensorial quantities
#latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)
Aij = u[i].dx(j)
A = as_tensor(Aij, (i,j))

def X(z):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([x[0], x[1], z])

#e(z)[i] = e_i_{al-izzi2020shear}
def e(z):
    return as_tensor([[1, 0, z.dx(0)], [0, 1, z.dx(1)]])

#normal(z) = \hat{n}_{al-izzi2020shear}
def normal(z):
    return as_tensor(cross(e(z)[0], e(z)[1]) /  ufl_norm(cross(e(z)[0], e(z)[1])) )


#b(z) = b_{ij}_{al-izzi2020shear}
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
def Nabla_v(v, z):
    return as_tensor((v[i]).dx(j) + v[k]*Gamma(z)[i, k, j], (i, j))

#covariant derivative of one-form omega with respect to \partial/partial x: Nabla_omega(omega, z)[i, j] = {\Nabla_j omega_i}_{al-izzi2020shear}
def Nabla_omega(omega, z):
    return as_tensor((omega[i]).dx(j) - omega[k]*Gamma(z)[k, i, j], (i, j))



# Define symmetric gradient
def epsilon(u):
    # nabla_grad(u)_{i,j} = (u[j]).dx[i]
    #sym(nabla_grad(u)) =  nabla_grad(u)_{i,j} + nabla_grad(u)_{j,i}
    # return sym(nabla_grad(u))
    return as_tensor(0.5*(u[i].dx(j) + u[j].dx(i)), (i,j))

# Define stress tensor
def sigma(u, p):
    return as_tensor(2*epsilon(u)[i,j] - p*Identity(len(u))[i,j], (i, j))