from fenics import *
from mshr import *

import boundary_geometry as bgeo


'''
the auxiliary fields are defined as follows
- omega_n_12_i[i] == omega^{n-1/2}_i =  \partial_i  z^{n-1/2}
- mu_n_12 == mu^{n-1/2} = H(omega_n_12)
- nu_n_12[i] == nu^{n-1/2}_i = Nabla_i mu^{n-1/2} = Nabla_i H(omega_{n-1/2})
- tau_n_12 = Nabla^i nu^{n-1/2}_i
- d_n[i, j] = d_{ij}(V, W, omega_n_12) 
'''


degree_function_space = 1

# Define function spaces
#finite elements for sigma .... omega
P_v_bar = VectorElement( 'P', triangle, 2 )
P_w_bar = FiniteElement( 'P', triangle, 1 )
P_phi = FiniteElement('P', triangle, 1)
P_v_n = VectorElement( 'P', triangle, 2 )
P_w_n = FiniteElement( 'P', triangle, 1 )
P_z_n = FiniteElement( 'P', triangle, degree_function_space )
P_omega_n = VectorElement( 'P', triangle, degree_function_space )
P_mu_n = FiniteElement( 'P', triangle, degree_function_space )

element = MixedElement( [P_v_bar, P_w_bar, P_phi, P_v_n, P_w_n, P_z_n, P_omega_n, P_mu_n] )
#total function space
Q = FunctionSpace(bgeo.mesh, element)
#function spaces for vbar .... zn
Q_v_bar = Q.sub(0).collapse()
Q_w_bar = Q.sub(1).collapse()
Q_phi = Q.sub(2).collapse()
Q_v_n = Q.sub(3).collapse()
Q_w_n = Q.sub(4).collapse()
Q_z_n= Q.sub(5).collapse()
Q_omega_n = Q.sub(6).collapse()
Q_mu_n = Q.sub(7).collapse()

#the function spaces for nu, tau and d are for post-processing only
Q_nu = VectorFunctionSpace( bgeo.mesh, 'P', degree_function_space )
Q_tau = FunctionSpace( bgeo.mesh, 'P', degree_function_space )
Q_d = TensorFunctionSpace( bgeo.mesh, 'P', degree_function_space, shape=(2, 2) )

# function space to store tangential force fields
Q_f_t = VectorFunctionSpace( bgeo.mesh, 'P', degree_function_space )
#function space to store normal force fields
Q_f_n = FunctionSpace( bgeo.mesh, 'P', degree_function_space )
Q_dFdl = VectorFunctionSpace( bgeo.mesh, 'P', degree_function_space )
Q_dFds = FunctionSpace( bgeo.mesh, 'P', degree_function_space )


# Define functions
#the Jacobian
J_psi = TrialFunction(Q)
psi = Function(Q)
nu_v_bar, nu_w_bar, nu_phi, nu_v_n, nu_w_n, nu_z_n_12, nu_omega_n_12, nu_mu_n_12 = TestFunctions( Q )
#fields at the preceeding steps
v_n_1 = Function(Q_v_n)
v_n_2 = Function(Q_v_n)
w_n_1 = Function(Q_w_n)
sigma_n_12 = Function( Q_phi )
sigma_n_32 = Function( Q_phi )
sigma_n_12_output = Function( Q_phi )
z_n_32 = Function( Q_z_n )

nu_n_12 = Function(Q_nu)
tau_n_12 = Function(Q_tau)
d = Function( Q_d )

J_pp_nu = TrialFunction( Q_nu )
J_pp_tau = TrialFunction( Q_tau )
J_pp_d = TrialFunction( Q_d )

nu_nu = TestFunction(Q_nu)
nu_tau = TestFunction(Q_tau)
nu_d = TestFunction(Q_d)


#these functions are used to print the solution to file
#     v_bar_output, w_bar_output, phi_output, v_n_output, w_n_output, omega_n_12_output, z_n_12_output = psi.split( deepcopy=True )
v_bar_output= Function(Q_v_bar)
w_bar_output = Function(Q_w_bar)
phi_output = Function(Q_phi)
v_n_output = Function(Q_v_n)
w_n_output = Function(Q_w_n)
z_n_12_output = Function(Q_z_n)
omega_n_12_output = Function(Q_omega_n)
mu_n_output = Function(Q_mu_n)

#vbar_0, ...., z_n_0 are used to store the initial conditions
v_bar_0 = Function( Q_v_bar )
w_bar_0 = Function( Q_w_bar )
phi_0 = Function(Q_phi)
# sigma_n_12_0 = Function( Q_phi )
v_n_0 = Function( Q_v_n )
w_n_0 = Function( Q_w_n )
z_n_12_0 = Function( Q_z_n )
omega_n_12_0 = Function( Q_omega_n )
mu_n_12_0 = Function( Q_mu_n )

nu_0 = Function( Q_nu )
tau_0 = Function( Q_tau )
d_0 = Function( Q_d )

v_bar, w_bar, phi, v_n, w_n, z_n_12, omega_n_12, mu_n_12 = split( psi )
V = (v_bar + v_n_1) / 2.0
W = (w_bar + w_n_1) / 2.0

#fields to store tangential 't' and normal 'n' forces
#f_visc_t[i] = f^{VISC i}(V, W, omega^{n-1/2})_notes
f_visc_t = Function(Q_f_t)
f_sigma_t = Function(Q_f_t)
f_v_t = Function(Q_f_t)

f_visc_n = Function(Q_f_n)
f_el_n = Function(Q_f_n)
f_laplace = Function( Q_f_n )

#field to store the tangential force per unit length exerted on a line element: dFdl[i] = dF^i/dl _notes
dFdl = Function( Q_dFdl )
#field to store the normal force per unit surface exerted on a surface element: dFds = dF_n/dS _notes
dFds = Function( Q_dFds )


#fields to store the residual of variational problems
# res_F_v_bar = Function( Q_v_bar )
res_F_omega_n = Function( Q_f_n )


# assigner = FunctionAssigner(Q, [Q_v_bar, Q_w_bar, Q_phi, Q_v_n, Q_w_n, Q_z_n, Q_omega_n, Q_mu_n])
# assigner.assign(psi, [v_bar_0, w_bar_0, phi_0, v_n_0, w_n_0, z_n_0, omega_n_0, mu_n_0])
