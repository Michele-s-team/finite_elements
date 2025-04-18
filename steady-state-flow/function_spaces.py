from fenics import *
from mshr import *

import boundary_geometry as bgeo


degree_function_space = 1

# Define function spaces
#finite elements for sigma .... omega
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_sigma_n = FiniteElement( 'P', triangle, 1 )
P_z_n = FiniteElement( 'P', triangle, degree_function_space )
P_omega_n = VectorElement( 'P', triangle, degree_function_space )
P_mu_n = FiniteElement( 'P', triangle, degree_function_space )


element = MixedElement( [P_v_n, P_w_n, P_sigma_n, P_z_n, P_omega_n, P_mu_n] )
#total function space
Q = FunctionSpace(bgeo.mesh, element)
#function spaces for vbar .... zn
Q_v = Q.sub( 0 ).collapse()
Q_w = Q.sub( 1 ).collapse()
Q_sigma = Q.sub( 2 ).collapse()
Q_z= Q.sub( 3 ).collapse()
Q_omega = Q.sub( 4 ).collapse()
Q_mu = Q.sub( 5 ).collapse()

#the function spaces for nu, tau and d are for post-processing only
Q_nu = VectorFunctionSpace( bgeo.mesh, 'P', degree_function_space )
Q_tau = FunctionSpace( bgeo.mesh, 'P', degree_function_space )
Q_d = TensorFunctionSpace( bgeo.mesh, 'P', degree_function_space, shape=(2, 2) )

# function space to store tangential force fields
Q_f_t = VectorFunctionSpace( bgeo.mesh, 'P', degree_function_space )
#function space to store normal force fields
Q_f_n = FunctionSpace( bgeo.mesh, 'P', degree_function_space )

# function spaces for the tangential and normal forces per unit length
Q_dFfl_t = VectorFunctionSpace(bgeo.mesh, 'P', degree_function_space)
Q_dFfl_n = FunctionSpace(bgeo.mesh, 'P', degree_function_space)

#function space for three-dimensional vector fields depending on the two coordinates on the mesh
Q_3d = VectorFunctionSpace( bgeo.mesh, 'P', degree_function_space, dim=3 )


'''
function spaces of polynomial order 1 (which should not be changed) which are used to read in functions and assign their nodal values from a list 
as in function.set_from_list and function.set_from_file
'''
Q_read = FunctionSpace( bgeo.mesh, 'P', 1 )


# Define functions
J_psi = TrialFunction( Q )
psi = Function( Q )
nu_v, nu_w, nu_sigma,  nu_z, nu_omega, nu_mu = TestFunctions( Q )

nu = Function(Q_nu)
tau = Function(Q_tau)
d = Function( Q_d )

J_pp_nu = TrialFunction( Q_nu )
J_pp_tau = TrialFunction( Q_tau )
J_pp_d = TrialFunction( Q_d )

nu_nu = TestFunction(Q_nu)
nu_tau = TestFunction(Q_tau)
nu_d = TestFunction(Q_d)


#these functions are used to print the solution to file
v_output = Function(Q_v)
w_output = Function(Q_w)
sigma_output = Function(Q_sigma)
z_output = Function(Q_z)
omega_output = Function(Q_omega)
mu_output = Function(Q_mu)

#functions used to store the nodal values read from a list or file
v_0_r_read = Function( Q_read )
w_0_read = Function( Q_read )
sigma_0_read = Function( Q_read )
z_0_read = Function( Q_read )
omega_0_r_read = Function( Q_read )
mu_0_read = Function( Q_read )

# v_0, .... are used to store the initial conditions
v_0 = Function( Q_v )
w_0 = Function( Q_w )
sigma_0 = Function( Q_sigma )
z_0 = Function( Q_z )
omega_0 = Function( Q_omega )
mu_0 = Function( Q_mu )

nu_0 = Function( Q_nu )
tau_0 = Function( Q_tau )
d_0 = Function( Q_d )

v, w, sigma, z, omega, mu = split( psi )
assigner = FunctionAssigner(Q, [Q_v, Q_w, Q_sigma, Q_z, Q_omega, Q_mu])