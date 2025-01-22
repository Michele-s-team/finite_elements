from fenics import *
from mshr import *
import ufl as ufl
from dolfin import *
import numpy as np


#read the mesh
mesh = Mesh()
xdmf = XDMFFile(mesh.mpi_comm(),  "triangle_mesh.xdmf")
xdmf.read(mesh)

#read the triangles
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile( "triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#read the lines
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile( "line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()

#radius of the smallest cell in the mesh
r_mesh = mesh.hmin()
r = 1.0
R = 2.0
c_r = [0, 0]
c_R = [0, 0]


# norm of vector x
def my_norm(x):
    return (sqrt(np.dot(x, x)))

#this is the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega
facet_normal = FacetNormal( mesh )

# test for surface elements
dx = Measure( "dx", domain=mesh, subdomain_data=sf, subdomain_id=1 )
ds_r = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=2 )
ds_R = Measure( "ds", domain=mesh, subdomain_data=mf, subdomain_id=3 )


# Define boundaries and obstacle
boundary_r = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) < (1.0 + 2.0)/2.0'
boundary_R = 'on_boundary && sqrt(pow(x[0], 2) + pow(x[1], 2)) > (1.0 + 2.0)/2.0'

'''
The fields z, omega, mu and nu are defined as follows: 
z
omega_i = \partial_i z
mu = H(omega)
nu_i = Nabla_i mu (where Nabla is the covariant derivative)
'''

degree_function_space = 2
P_z = FiniteElement( 'P', triangle, degree_function_space )
P_omega = VectorElement( 'P', triangle, degree_function_space )
P_mu = FiniteElement( 'P', triangle, degree_function_space )
P_nu = VectorElement( 'P', triangle, degree_function_space )

element = MixedElement( [P_z, P_omega, P_mu, P_nu] )
Q = FunctionSpace(mesh, element)
Q_z= Q.sub( 0 ).collapse()
Q_omega = Q.sub( 1 ).collapse()
Q_mu = Q.sub( 2 ).collapse()
Q_nu = Q.sub( 3 ).collapse()
Q_sigma = FunctionSpace( mesh, 'P', 1 )

J_psi = TrialFunction( Q )
psi = Function( Q )
nu_z, nu_omega, nu_mu, nu_nu = TestFunctions( Q )

#these functions are used to print the solution to file
sigma = Function(Q_sigma)
z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function( Q_mu )
nu_output = Function( Q_nu )

z, omega, mu, nu = split( psi )

def ufl_norm(x):
    return(sqrt(ufl.dot(x, x)))

i, j, k, l = ufl.indices(4)

def X(z):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([x[0], x[1], z])

def e(omega):
    return as_tensor([[1, 0, omega[0]], [0, 1, omega[1]]])

def normal(omega):
    return as_tensor(cross(e(omega)[0], e(omega)[1]) /  ufl_norm(cross(e(omega)[0], e(omega)[1])) )

def b(omega):
    return as_tensor((normal(omega))[k] * (e(omega)[i, k]).dx(j), (i,j))

def g(omega):
    return as_tensor([[1+ (omega[0])**2, (omega[0])*(omega[1])],[(omega[0])*(omega[1]), 1+ (omega[1])**2]])

def g_c(omega):
    return ufl.inv(g(omega))

def detg(omega):
    return ufl.det(g(omega))

def sqrt_detg(omega):
    return sqrt(detg(omega))

def sqrt_abs_detg(omega):
    return sqrt(abs_detg(omega))

def dydtheta(c):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([-(x[1]-c[1]), x[0]-c[0]])

def sqrt_deth_circle(omega, c):
    return(sqrt((dydtheta(c))[i]*(dydtheta(c))[j]*g(omega)[i, j]))


#normal vectors
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

def Nt_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return as_tensor(g_c(omega)[i, j] * N3d[k] * e(omega)[j, k], (i))

def Nn_circle(omega):
    N3d = as_tensor([facet_normal[0], facet_normal[1], 0.0])
    return (N3d[i] * (normal(omega))[i])

def n_circle(omega):
    return as_tensor((Nt_circle(omega))[k] / sqrt(g(omega)[i, j]* (Nt_circle(omega))[i] *  (Nt_circle(omega))[j] ), (k))


#other geometrical quantitites
def H(omega):
    return (0.5 * g_c(omega)[i, j]*b(omega)[j, i])

def K(omega):
    return(ufl.det(as_tensor(b(omega)[i,k]*g_c(omega)[k,j], (i, j))))

def Gamma(omega):
    return as_tensor(0.5 * g_c(omega)[i,l] * ( (g(omega)[l, k]).dx(j) + (g(omega)[j, l]).dx(k) - (g(omega)[j, k]).dx(l) ), (i, j, k))

def Nabla_v(u, omega):
    return as_tensor((u[i]).dx(j) + u[k] * Gamma(omega)[i, k, j], (i, j))

def Nabla_f(f, omega):
    return as_tensor((f[i]).dx(j) - f[k] * Gamma(omega)[k, i, j], (i, j))

def fel_n(omega, mu, nu, kappa):
    return (kappa * ( - 2.0 * g_c(omega)[i, j] * Nabla_f(nu, omega)[i, j] - 4.0 * mu * ( (mu**2) - K(omega) ) ))

def flaplace(sigma, omega):
    return (2.0 * sigma * H(omega))


# model parameters
kappa = 1.0
rho = 1.0
eta = 1.0
C = 0.1
alpha = 1e1


z_r_const = 1.0/10.0
z_R_const = 4.0/5.0
zp_r_const = -3.0/10.0
zp_R_const = 6.0/5.0
omega_r_const = - r * zp_r_const / sqrt( r**2  * (1.0 + zp_r_const**2))
omega_R_const = R * zp_R_const / sqrt( R**2  * (1.0 + zp_R_const**2))


class sigma_Expression( UserExpression ):
    def eval(self, values, x):
        # values[0] = (2.0 + C**2) * kappa / (2.0 * (1.0 + C**2) * (x[0]**2 + x[1]**2))
        values[0] =  cos(2.0*(np.pi)*my_norm(x))

    def value_shape(self):
        return (1,)

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


# values of omega for the BCs
omega_r = interpolate( omega_r_Expression( element=Q_z.ufl_element() ), Q_z )
omega_R = interpolate( omega_R_Expression( element=Q_z.ufl_element() ), Q_z )
sigma.interpolate( sigma_Expression( element=Q_sigma.ufl_element() ) )

# boundary conditions
bc_z_r = DirichletBC( Q.sub( 0 ), Expression( 'z_r', z_r=z_r_const, element=Q.sub( 0 ).ufl_element() ), boundary_r )
bc_z_R = DirichletBC( Q.sub( 0 ), Expression( 'z_R', z_R=z_R_const, element=Q.sub( 0 ).ufl_element() ), boundary_R )
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

F = (F_z + F_omega + F_mu + F_nu) + F_N

dolfin.parameters["form_compiler"]["quadrature_degree"] = 10

kappa = Constant( kappa )

#solve the problem
J = derivative( F, psi, J_psi )
problem = NonlinearVariationalProblem( F, psi, bcs, J )
solver = NonlinearVariationalSolver( problem )
solver.solve()


# Create XDMF files for visualization output
xdmffile_z = XDMFFile( 'z.xdmf' )
xdmffile_omega = XDMFFile( 'omega.xdmf' )
xdmffile_mu = XDMFFile( 'mu.xdmf' )
xdmffile_nu = XDMFFile( 'nu.xdmf' )
xdmffile_sigma = XDMFFile( 'sigma.xdmf' )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output = psi.split( deepcopy=True )

# print solution to file
xdmffile_z.write( z_output, 0 )
xdmffile_omega.write( omega_output, 0 )
xdmffile_mu.write( mu_output, 0 )
xdmffile_nu.write( nu_output, 0 )
xdmffile_sigma.write( sigma, 0 )


xdmffile_check = XDMFFile( "check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

print("Check if the PDE is satisfied:")
print( "\t<<residual_of_the_equation^2>> = ", \
   sqrt(assemble( ( (  fel_n( omega_output, mu_output, nu_output, kappa ) + flaplace( sigma, omega_output) ) ** 2 * dx ) ) / assemble(Constant(1.0) * dx))
  )
xdmffile_check.write( project( fel_n( omega_output, mu_output, nu_output, kappa ) + flaplace( sigma, omega_output) , Q_sigma ), 0 )
xdmffile_check.close()