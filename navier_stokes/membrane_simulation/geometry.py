import numpy as np
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

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)


# Define boundaries
#a semi-circle given by the left half of circle_R
inflow   = 'on_boundary && (x[0] < 0.01) && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#a semi-circle given by the right half of circle_R
outflow   =  'on_boundary && (x[0] > 0.01) && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#the whole circle_R
external_boundary = 'on_boundary && (x[0]*x[0] + x[1]*x[1] > (0.5*0.5))'
#the obstacle
cylinder = 'on_boundary && (x[0]*x[0] + x[1]*x[1] < (0.5*0.5))'


# Define norm of x
def norm(x):
    return (np.sqrt(np.dot(x, x)))



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
        # values[0] = sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
        values[0] = sin(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
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
i, j = ufl.indices(2)
Aij = u[i].dx(j)
A = as_tensor(Aij, (i,j))

def X(z):
    return as_tensor([x[0], x[1], z(x)])

#e_i = \partial_i X
def e(z):
    return as_tensor(z.dx(i), (i,))

#g_{ij}
def g(z):
    return as_tensor(dot((e(z))[i], (e(z))[j]), (i, j))

# Define symmetric gradient
def epsilon(u):
    # nabla_grad(u)_{i,j} = (u[j]).dx[i]
    #sym(nabla_grad(u)) =  nabla_grad(u)_{i,j} + nabla_grad(u)_{j,i}
    # return sym(nabla_grad(u))
    return as_tensor(0.5*(u[i].dx(j) + u[j].dx(i)), (i,j))

# Define stress tensor
def sigma(u, p):
    return as_tensor(2*epsilon(u)[i,j] - p*Identity(len(u))[i,j], (i, j))