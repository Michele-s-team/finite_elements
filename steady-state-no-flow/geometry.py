from fenics import *
from mshr import *
import ufl as ufl
from function_spaces import *


#norm for UFL vectors
def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))

epsilon = ufl.PermutationSymbol(2)


#definition of scalar, vectorial and tensorial quantities
#latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)

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

#square root of the determinant of the pull-back of the metric, h, on a circle with radius r centered at c ( it is independent of r). This pull-back is done by parameterizing the circle, \partial \Omega_O witht the polar angle \theta as a variable
def sqrt_deth_circle(omega, c):
    return(sqrt((dydtheta(c))[i]*(dydtheta(c))[j]*g(omega)[i, j]))

#square root of the determinant of the pull-back of the metric on \partial \Omega_in(out), parametrized with l , given by  x^1 = 0 (L) and x^2 = l, as coordinate for \partial \Omega_in (out)
def sqrt_deth_lr(omega):
    return sqrt(g(omega)[1,1])

#square root of the determinant of the pull-back of the metric on \partial \Omega_W (top or bottom), parametrized with l , given by  x^1 = l and x^2 = 0 (h), as coordinate for \partial \Omega_W
def sqrt_deth_tb(omega):
    return sqrt(g(omega)[0,0])

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


#Nt^i_notes on \partial \Omega_in and out
def Nt_lr(omega):
    x = ufl.SpatialCoordinate(mesh)
    N3d = as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0, 0.0] )
    return as_tensor(g_c(omega)[i, j] * N3d[k] * e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_in and out
def Nn_lr(omega):
    x = ufl.SpatialCoordinate(mesh)
    N3d = as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0, 0.0] )
    return (N3d[i] * (normal(omega))[i])

#Nt^i_notes on \partisal \Omega_W
def Nt_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    N3d = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0), 0.0] )
    return as_tensor(g_c(omega)[i, j] * N3d[k] * e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_top and bottom
def Nn_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    N3d = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0), 0.0] )
    return (N3d[i] * (normal(omega))[i])

#Nt^i_notes on \partisal \Omega_O
def Nt_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return as_tensor(g_c(omega)[i, j] * N3d[k] * e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_O
def Nn_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return (N3d[i] * (normal(omega))[i])

#n^i_notes on \partial \Omega_in and out
def n_lr(omega):
    return as_tensor((Nt_lr(omega))[k] / sqrt(g(omega)[i, j]* (Nt_lr(omega))[i] *  (Nt_lr(omega))[j] ), (k))

def n_tb(omega):
    return as_tensor((Nt_tb(omega))[k] / sqrt(g(omega)[i, j]* (Nt_tb(omega))[i] *  (Nt_tb(omega))[j] ), (k))

def n_circle(omega):
    return as_tensor((Nt_circle(omega))[k] / sqrt(g(omega)[i, j]* (Nt_circle(omega))[i] *  (Nt_circle(omega))[j] ), (k))


#the normal to the manifold pointing outwards the manifold and normalized according to the Euclidean metric, which can be plotted as a field
def facet_normal_smooth():
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

#lalplace-beltrami operator applied to a scalar function f : Nabla_LB(f, omega) = \Nabla_{LB} f_notes
def Nabla_LB(f, omega):
    return (- 1.0/sqrt_detg(omega) * ( ( sqrt_detg(omega) * g_c(omega)[i, j] * (f.dx(j)) ).dx(i) ) )

#2-covariant rate-of_deformation tensor for zero normal velocity: d(u, z)[i, j] = {d_{ij}}_{alizzi2020shear for zero w}
def d(v, w, omega):
    return as_tensor( 0.5 * (g(omega)[i, k] * Nabla_v(v, omega)[k, j] + g(omega)[j, k] * Nabla_v(v, omega)[k, i]) - (b(omega)[i,j]) * w, (i, j) )

#2-contravariant rate-of_deformation tensor: d_c(u, un, z)[i, j] = {d^{ij}}_{alizzi2020shear}
def d_c(v, w, omega):
    return as_tensor( g_c(omega)[i, k] * g_c(omega)[j, l] * d(v, w, omega)[k,l], (i, j) )

#given a vector and a scalar, return the vector vector^i * scalar
def vector_times_scalar(vector, scalar):
    return as_tensor(scalar * vector[i], (i))