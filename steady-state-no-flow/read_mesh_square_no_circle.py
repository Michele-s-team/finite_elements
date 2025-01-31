from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import ufl as ufl

import runtime_arguments as rarg
import mesh as msh

#read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), (rarg.args.input_directory) + "/triangle_mesh.xdmf")
xdmf.read(mesh)

import geometry as geo
i, j, k, l = ufl.indices(4)


#Nt^i_notes on \partisal \Omega_O
def Nt_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_O
def Nn_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return (N3d[i] * (normal(omega))[i])

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
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_in and out
def Nn_lr(omega):
    x = ufl.SpatialCoordinate(mesh)
    N3d = as_tensor([conditional(lt(x[0], L/2.0), -1.0, 1.0), 0.0, 0.0] )
    return (N3d[i] * (normal(omega))[i])

#Nt^i_notes on \partisal \Omega_W
def Nt_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    N3d = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0), 0.0] )
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))

#N_n_notes on \partial \Omega_top and bottom
def Nn_tb(omega):
    x = ufl.SpatialCoordinate(mesh)
    N3d = as_tensor([0.0, conditional(lt(x[1], h/2.0), -1.0, 1.0), 0.0] )
    return (N3d[i] * (normal(omega))[i])


#n^i_notes on \partial \Omega_in and out
def n_lr(omega):
    return as_tensor((Nt_lr(omega))[k] / sqrt(geo.g(omega)[i, j]* (Nt_lr(omega))[i] *  (Nt_lr(omega))[j] ), (k))

def n_tb(omega):
    return as_tensor((Nt_tb(omega))[k] / sqrt(geo.g(omega)[i, j]* (Nt_tb(omega))[i] *  (Nt_tb(omega))[j] ), (k))

def n_circle(omega):
    return as_tensor((Nt_circle(omega))[k] / sqrt(geo.g(omega)[i, j]* (Nt_circle(omega))[i] *  (Nt_circle(omega))[j] ), (k))

#the normal to the manifold pointing outwards the manifold and normalized according to the Euclidean metric, which can be plotted as a field
def facet_normal_smooth():
    u = calc_normal_cg2(mesh)
    return as_tensor(u[k], (k))


#read the triangles
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the lines
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#radius of the smallest cell in the mesh
r_mesh = mesh.hmin()


#CHANGE PARAMETERS HERE
L = 1.0
h = 1.0
#CHANGE PARAMETERS HERE


#this is the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega
facet_normal = FacetNormal( mesh )

# read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet( mesh, mvc )

# test for surface elements
ds_l = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=2 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=3 )
ds_t = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=4 )
ds_b = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=5 )


#a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

# here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
integral_l = assemble( f_test_ds * ds_l )
integral_r = assemble( f_test_ds * ds_r )
integral_t = assemble( f_test_ds * ds_t )
integral_b = assemble( f_test_ds * ds_b )

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print( "Integral l = ", integral_l, " exact value = 0.962047" )
print( "Integral r = ", integral_r, " exact value = 0.996086" )
print( "Integral t = ", integral_t, " exact value = 5.26416" )
print( "Integral b = ", integral_b, " exact value = 4.97026" )

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_l  = 'near(x[0], 0.0)'
boundary_r  = 'near(x[0], 0.5)'
boundary_t  = 'near(x[1], 0.5)'
boundary_b  = 'near(x[1], 0.0)'
boundary_lr  = 'near(x[0], 0) || near(x[0], 0.5)'
boundary_tb  = 'near(x[1], 0) || near(x[1], 0.5)'
boundary_square = 'on_boundary'
#CHANGE PARAMETERS HERE