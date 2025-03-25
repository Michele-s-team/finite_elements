from fenics import *
from dolfin import *
from mshr import *
import ufl as ufl


import geometry as geo
import mesh as mesh_module
import runtime_arguments as rarg


#read the mesh
mesh = mesh_module.read_mesh(rarg.args.input_directory + "/triangle_mesh.xdmf")


#the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega
facet_normal = FacetNormal( mesh )


i, j, k, l = ufl.indices(4)

#3D normal vector to the mesh boundary 
def N_3D():
    return as_tensor([facet_normal[0], facet_normal[1], 0.0])


#N_n_notes on \partial \Omega_O
def Nn_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return (N3d[i] * (geo.n(omega))[i])

#vector used to define the pull-back of the metric, h, on a circle with radius r centered at c ( it is independent of r), see 'notes reall2013general'
def dydtheta(c):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([-(x[1]-c[1]), x[0]-c[0]])

#square root of the determinant of the pull-back of the metric, h, on a circle with radius r centered at c ( it is independent of r). This pull-back is done by parameterizing the circle, \partial \Omega_O witht the polar angle \theta as a variable
def sqrt_deth_circle(omega, c):
    return(sqrt((dydtheta(c))[i]*(dydtheta(c))[j]*geo.g(omega)[i, j]))

#square root of the determinant of the pull-back of the metric on \partial \Omega_in(out), parametrized with l , given by  x^1 = 0 (L) and x^2 = l, as coordinate for \partial \Omega_in (out)
def sqrt_deth_lr(omega):
    return sqrt(geo.g(omega)[1,1])

#square root of the determinant of the pull-back of the metric on \partial \Omega_W (top or bottom), parametrized with l , given by  x^1 = l and x^2 = 0 (h), as coordinate for \partial \Omega_W
def sqrt_deth_tb(omega):
    return sqrt(geo.g(omega)[0,0])

#This function extend the facet normal to the interior of the cells and project into the tangent space (used for visualization purposes)
def calc_normal_cg2(omega, mesh):
    n = FacetNormal(mesh)
    n = geo.norm_t(omega, geo.from_3D_to_t(omega, as_tensor([n[0], n[1], 0.0])))
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
    L = (mesh_module.extremal_coordinates( mesh ))[0][1]

    N3d = as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0, 0.0] )
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_in and out
def Nn_lr(omega):
    x = ufl.SpatialCoordinate(mesh)
    L = (mesh_module.extremal_coordinates( mesh ))[0][1]

    N3d = as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0, 0.0] )
    return (N3d[i] * (geo.n(omega))[i])

#Nt^i_notes on \partisal \Omega_W
def Nt_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    h = (mesh_module.extremal_coordinates( mesh ))[1][1]

    N3d = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0), 0.0] )
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_top and bottom
def Nn_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    h = (mesh_module.extremal_coordinates( mesh ))[1][1]

    N3d = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0), 0.0] )
    return (N3d[i] * (geo.n(omega))[i])

#n^i_notes on \partial \Omega_in and out
def n_lr(omega):
    return geo.norm_t(omega, geo.from_3D_to_t(omega, N_3D()))

def n_tb(omega):
    return geo.norm_t(omega, geo.from_3D_to_t(omega, N_3D()))

def n_circle(omega):
    return geo.norm_t(omega, geo.from_3D_to_t(omega, N_3D()))

def n_c(omega):
    return 0.01*as_tensor([facet_normal[0], facet_normal[1]])

#tangent vector to the boundary curve in the tangent space
def tang_t(omega):
    return geo.from_3D_to_t(omega, as_tensor([-facet_normal[1], facet_normal[0], 0.0]))


#the normal to the manifold pointing outwards the manifold and normalized according to the Euclidean metric, which can be plotted as a field
def facet_normal_smooth():
    u = calc_normal_cg2(mesh)
    return as_tensor(u[k], (k))

