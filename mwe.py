'''

Run with
clear; clear; rm -r solution; python3 mwe.py /home/fenics/shared/mesh /home/fenics/shared/solution

The solution files will be stored in /home/fenics/shared/solution

Note that all sections of the code which need to be changed when an external parameter (e.g. the length of the Rectangle, etc...) is changed are bracketed by
#CHANGE PARAMETERS HERE
'''

from __future__ import print_function
from fenics import *
from mshr import *
import ufl as ufl
import argparse
from dolfin import *

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

#read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), (args.input_directory) + "/triangle_mesh.xdmf")
xdmf.read(mesh)

#read the triangles
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the lines
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()


# # Create mesh
# channel = Rectangle(Point(0, 0), Point(1.0, 1.0))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# mesh = generate_mesh(domain, 64)

#radius of the smallest cell in the mesh
r_mesh = mesh.hmin()

#CHANGE PARAMETERS HERE
r = 1.0
R = 2.0
c_r = [0, 0]
c_R = [0, 0]

tol = 1E-3
#CHANGE PARAMETERS HERE


# norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))

#this is the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega
facet_normal = FacetNormal( mesh )

# test for surface elements
dx = Measure( "dx", domain=mesh, subdomain_data=sf, subdomain_id=1 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=2 )
ds_R = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=3 )


#a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralsds(UserExpression):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = cos(my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegralsds( element=Q_test.ufl_element() ) )

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
numerical_value_int_dx = assemble( f_test_ds * dx )
exact_value_int_dx = 2.90212
print(f"\int_box_minus_ball f dx = {numerical_value_int_dx}, should be  {exact_value_int_dx}, relative error =  {abs( (numerical_value_int_dx - exact_value_int_dx) / exact_value_int_dx ):e}" )

exact_value_int_ds_r = 2.77595
numerical_value_int_ds_r = assemble( f_test_ds * ds_r )
print(f"\int_sphere f ds = {numerical_value_int_ds_r}, should be  {exact_value_int_ds_r}, relative error =  {abs( (numerical_value_int_ds_r - exact_value_int_ds_r) / exact_value_int_ds_r ):e}" )

exact_value_int_ds_R = 3.67175
numerical_value_int_ds_R = assemble( f_test_ds * ds_R )
print(f"\int_sphere f ds = {numerical_value_int_ds_R}, should be  {exact_value_int_ds_R}, relative error =  {abs( (numerical_value_int_ds_R - exact_value_int_ds_R) / exact_value_int_ds_R ):e}" )



# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
boundary = 'on_boundary'
boundary_r = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) < (1.0 + 2.0)/2.0'
boundary_R = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) > (1.0 + 2.0)/2.0'
#CHANGE PARAMETERS HERE


#this function prints a scalar field to csv file
def print_scalar_to_csvfile(f, filename):
    csvfile = open(filename, "w")
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )
    for x, val in zip(f.function_space().tabulate_dof_coordinates(), f.vector().get_local()):
        print(f"{val},{x[0]},{x[1]},{0}", file=csvfile)
    csvfile.close()


#this function prints a vector field to csv file
def print_vector_to_csvfile(f, filename):

    i=0
    list_val_x = []
    list_val_y = []
    list_x = []
    for x, val in zip(f.function_space().tabulate_dof_coordinates(), f.vector().get_local()):
        if(i % 2 == 0):
            list_val_x.append(val)
            list_x.append( x )
        else:
            list_val_y.append(val)

        i += 1

    csvfile = open(filename, "w")
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for x, val_x, val_y in zip(list_x, list_val_x, list_val_y):
        print(f"{val_x},{val_y},{0},{x[0]},{x[1]},{0}", file=csvfile)

    csvfile.close()

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

#Pi(v, w, omega, sigma)[i, j] = \Pi^{ij}_notes, i.e., the momentum-flux tensor
def Pi(v, w, omega, sigma, eta):
    return as_tensor( - g_c(omega)[i, j] * sigma - 2.0 * eta * d_c(v, w, omega)[i ,j], (i, j) )

#dFdl(v, w, omega, sigma, eta, nu)[i] = dF^i/dl_notes, i.e., the force per unit length exerted on a line element with normal nu[i] = \nu^i_notes
def dFdl(v, w, omega, sigma, eta, nu):
    return as_tensor(Pi(v, w, omega, sigma, eta)[i, j] * g(omega)[j, k] * nu[k], (i))

#fel_n = f^{EL}_notes , i.e.,  part of the normal force due to the bending rigidity
def fel_n(omega, mu, nu, kappa):
    return (kappa * ( - 2.0 * g_c(omega)[i, j] * Nabla_f(nu, omega)[i, j] - 4.0 * mu * ( (mu**2) - K(omega) ) ))

#fvisc_n(v, w, omega, eta) = f^{VISC}_n_notes, i.e., viscous contribution to the normal force
def fvisc_n(v, w, omega, eta):
    return ( 2.0 * eta * ( g_c(omega)[i, k] * Nabla_v(v, omega)[j, k] * b(omega)[i, j] - 2.0 * w * ( 2.0 * ((H(omega))**2) - K(omega) )  )  )

#tforce coming from the Laplace preccure
def flaplace(sigma, omega):
    return (2.0 * sigma * H(omega))

# CHANGE PARAMETERS HERE
# bending rigidity
kappa = 1.0
# density
rho = 1.0
# viscosity
eta = 1.0
C = 0.1
#values of z at the boundaries
'''
if you compare with the solution from check-with-analytical-solution-bc-ring.nb:
    - z_r(R)_const_{here} <-> zRmin(max)_{check-with-analytical-solution-bc-ring.nb}
    - zp_r(R)_const_{here} <-> zpRmin(max)_{check-with-analytical-solution-bc-ring.nb}
'''
z_r_const = 1.0/10.0
z_R_const = 4.0/5.0
zp_r_const = -3.0/10.0
zp_R_const = 6.0/5.0
omega_r_const = - r * zp_r_const / sqrt( r**2  * (1.0 + zp_r_const**2))
omega_R_const = R * zp_R_const / sqrt( R**2  * (1.0 + zp_R_const**2))
# Nitche's parameter
alpha = 1e1


class SurfaceTensionExpression( UserExpression ):
    def eval(self, values, x):
        # values[0] = (2.0 + C**2) * kappa / (2.0 * (1.0 + C**2) * (x[0]**2 + x[1]**2))
        values[0] =  cos(2.0*(np.pi)*my_norm(x))

    def value_shape(self):
        return (1,)


class ManifoldExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class OmegaExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class MuExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0

    def value_shape(self):
        return (1,)


class NuExpression( UserExpression ):
    def eval(self, values, x):
        values[0] = 0
        values[1] = 0

    def value_shape(self):
        return (2,)


class omega_r_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_r_const

    def value_shape(self):
        return (1,)


class omega_R_Expression( UserExpression ):
    def eval(self, values, x):
        values[0] = omega_R_const

    def value_shape(self):
        return (1,)


# CHANGE PARAMETERS HERE


# the values of \partial_i z = omega_i on the circle and on the square, to be used in the boundary conditions (BCs) imposed with Nitche's method, in F_N
omega_r = interpolate( omega_r_Expression( element=Q_z.ufl_element() ), Q_z )
omega_R = interpolate( omega_R_Expression( element=Q_z.ufl_element() ), Q_z )

sigma.interpolate( SurfaceTensionExpression( element=Q_sigma.ufl_element() ) )
z_0.interpolate( ManifoldExpression( element=Q_z.ufl_element() ) )
omega_0.interpolate( OmegaExpression( element=Q_omega.ufl_element() ) )
mu_0.interpolate( MuExpression( element=Q_mu.ufl_element() ) )
nu_0.interpolate( NuExpression( element=Q_nu.ufl_element() ) )

# uncomment this if you want to assign to psi the initial profiles stored in v_0, ..., z_0
# assigner.assign(psi, [z_0, omega_0, mu_0, nu_0])

# boundary conditions (BCs)

# CHANGE PARAMETERS HERE
# BCs for z
bc_z_r = DirichletBC( Q.sub( 0 ), Expression( 'z_r', z_r=z_r_const, element=Q.sub( 0 ).ufl_element() ), boundary_r )
bc_z_R = DirichletBC( Q.sub( 0 ), Expression( 'z_R', z_R=z_R_const, element=Q.sub( 0 ).ufl_element() ), boundary_R )
# CHANGE PARAMETERS HERE

# all BCs
bcs = [bc_z_r, bc_z_R]

# Define variational problem

F_z = (kappa * (g_c( omega )[i, j] * nu[j] * (nu_z.dx( i )) - 2.0 * mu * ((mu ** 2) - K( omega )) * nu_z) + sigma * mu * nu_z) * sqrt_detg( omega ) * dx \
      - ( \
                  + (kappa * (n_circle( omega ))[i] * nu_z * nu[i]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
                  + (kappa * (n_circle( omega ))[i] * nu_z * nu[i]) * sqrt_deth_circle( omega, c_R ) * (1.0 / R) * ds_R
      )

F_omega = (- z * Nabla_v( nu_omega, omega )[i, i] - omega[i] * nu_omega[i]) * sqrt_detg( omega ) * dx \
          + ((n_circle( omega ))[i] * g( omega )[i, j] * z * nu_omega[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
          + ((n_circle( omega ))[i] * g( omega )[i, j] * z * nu_omega[j]) * sqrt_deth_circle( omega, c_R ) * (1.0 / R) * ds_R

F_mu = ((H( omega ) - mu) * nu_mu) * sqrt_detg( omega ) * dx

F_nu = (nu[i] * nu_nu[i] + mu * Nabla_v( nu_nu, omega )[i, i]) * sqrt_detg( omega ) * dx \
       - ((n_circle( omega ))[i] * g( omega )[i, j] * mu * nu_nu[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
       - ((n_circle( omega ))[i] * g( omega )[i, j] * mu * nu_nu[j]) * sqrt_deth_circle( omega, c_r ) * (1.0 / R) * ds_R

F_N = alpha / r_mesh * ( \
            + (((n_circle( omega ))[i] * omega[i] - omega_r) * ((n_circle( omega ))[k] * g( omega )[k, l] * nu_omega[l])) * sqrt_deth_circle( omega, c_r ) * (1.0 / r) * ds_r \
            + (((n_circle( omega ))[i] * omega[i] - omega_R) * ((n_circle( omega ))[k] * g( omega )[k, l] * nu_omega[l])) * sqrt_deth_circle( omega, c_R ) * (1.0 / R) * ds_R \
    )

# total functional for the mixed problem
F = (F_z + F_omega + F_mu + F_nu) + F_N


set_log_level( 20 )
dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

print("Input diredtory = ", args.input_directory )
print("Output diredtory = ", args.output_directory )
print("Radius of mesh cell = ", r_mesh)

# Define expressions used in variational forms
kappa = Constant( kappa )

# solve the variational problem
J = derivative( F, psi, J_psi )
problem = NonlinearVariationalProblem( F, psi, bcs, J )
solver = NonlinearVariationalSolver( problem )


#set the solver parameters here
params = {'nonlinear_solver': 'newton',
           'newton_solver':
            {
                # 'linear_solver'           : 'superlu',
                'linear_solver'           : 'mumps',
                'absolute_tolerance'      : 1e-6,
                'relative_tolerance'      : 1e-6,
                'maximum_iterations'      : 1000000,
                'relaxation_parameter'    : 0.95,
             }
}
solver.parameters.update(params)

'''
#set the solver parameters here
params ={"newton_solver": {"linear_solver": 'superlu'}}
solver.parameters.update(params)
'''

solver.solve()


# Create XDMF files for visualization output
xdmffile_z = XDMFFile( (args.output_directory) + '/z.xdmf' )
xdmffile_omega = XDMFFile( (args.output_directory) + '/omega.xdmf' )
xdmffile_mu = XDMFFile( (args.output_directory) + '/mu.xdmf' )
xdmffile_nu = XDMFFile( (args.output_directory) + '/nu.xdmf' )

xdmffile_sigma = XDMFFile( (args.output_directory) + '/sigma.xdmf' )

xdmffile_n = XDMFFile( (args.output_directory) + '/n.xdmf' )
xdmffile_n.write( facet_normal_smooth(), 0 )

xdmffile_f = XDMFFile( (args.output_directory) + '/f.xdmf' )
xdmffile_f.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output = psi.split( deepcopy=True )

# print solution to file
xdmffile_z.write( z_output, 0 )
xdmffile_omega.write( omega_output, 0 )
xdmffile_mu.write( mu_output, 0 )
xdmffile_nu.write( nu_output, 0 )
xdmffile_sigma.write( sigma, 0 )

print_scalar_to_csvfile(z_output, (args.output_directory) + '/z.csv')
print_vector_to_csvfile(omega_output, (args.output_directory) + '/omega.csv')
print_scalar_to_csvfile(mu_output, (args.output_directory) + '/mu.csv')
print_vector_to_csvfile(nu_output, (args.output_directory) + '/nu.csv')
print_scalar_to_csvfile(sigma, (args.output_directory) + '/sigma.csv')


# write the solutions in .h5 format so it can be read from other codes
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/z.h5", "w" ).write( z_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/omega.h5", "w" ).write( omega_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/mu.h5", "w" ).write( mu_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/nu.h5", "w" ).write( nu_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/sigma.h5", "w" ).write( sigma, "/f" )

xdmffile_f.write( project(fel_n( omega_output, mu_output, nu_output, kappa ), Q_sigma), 0 )
xdmffile_f.write( project(flaplace( sigma, omega_output), Q_sigma), 0 )
# xdmffile_f.write( project(fvisc_n( v, w, omega_output, eta ), Q_omega), 0 )



xdmffile_check = XDMFFile( (args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )


# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output = psi.split( deepcopy=True )

print("Check of BCs:")
print( "\t<<(z - phi)^2>>_r = ", \
   sqrt(assemble( ( (z_output - z_r_const ) ** 2 * ds_r ) ) / assemble(Constant(1.0) * ds_r))
  )
print( "\t<<(z - phi)^2>>_R = ", \
   sqrt(assemble( ( (z_output - z_R_const ) ** 2 * ds_R ) ) / assemble(Constant(1.0) * ds_R))
  )
print( "\t<<(n^i \omega_i - psi )^2>>_r = ", \
   sqrt(assemble( ( ((n_circle( omega_output ))[i] * omega_output[i] - omega_r ) ** 2 * ds_r ) ) / assemble(Constant(1.0) * ds_r))
  )
print( "\t<<(n^i \omega_i - psi )^2>>_R = ", \
   sqrt(assemble( ( ((n_circle( omega_output ))[i] * omega_output[i] - omega_R ) ** 2 * ds_R ) ) / assemble( Constant(1.0) * ds_R))
  )

print("Check if the PDE is satisfied:")
print( "\t<<(fel + flaplace)^2>> = ", \
   sqrt(assemble( ( (  fel_n( omega_output, mu_output, nu_output, kappa ) + flaplace( sigma, omega_output) ) ** 2 * dx ) ) / assemble(Constant(1.0) * dx))
  )

xdmffile_check.write( project( fel_n( omega_output, mu_output, nu_output, kappa ) + flaplace( sigma, omega_output) , Q_sigma ), 0 )
xdmffile_check.close()

