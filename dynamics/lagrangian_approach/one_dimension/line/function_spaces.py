from fenics import *

import differential_geometry.boundary.geometry as bgeo
import mesh.load as lmsh
import parameters.read.solution as rpam




# Define function spaces
#finite elements for sigma .... omega
P_v_bar = VectorElement( 'P', interval, 2 )
P_w_bar = FiniteElement( 'P', interval, 1 )
P_phi = FiniteElement('P', interval, 1)
P_v_n = VectorElement( 'P', interval, 2 )
P_w_n = FiniteElement( 'P', interval, 1 )
P_u_n_12 = VectorElement('P', interval, rpam.parameters['function_space_degree'], dim=2)
P_nu_n_12 = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_psi_n_12 = FiniteElement('P', interval, rpam.parameters['function_space_degree'])
P_mu_n_12 = FiniteElement( 'P', interval, rpam.parameters['function_space_degree'] )

element = MixedElement( [P_v_bar, P_w_bar, P_phi, P_v_n, P_w_n, P_u_n_12, P_nu_n_12, P_psi_n_12, P_mu_n_12] )
#total function space
Q = FunctionSpace(lmsh.mesh, element)
#function spaces for vbar .... zn
Q_v_bar = Q.sub(0).collapse()
Q_w_bar = Q.sub(1).collapse()
Q_phi = Q.sub(2).collapse()
Q_v_n = Q.sub(3).collapse()
Q_w_n = Q.sub(4).collapse()
Q_u_n_12 = Q.sub(5).collapse()
Q_nu_n_12 = Q.sub(6).collapse()
Q_psi_n_12 = Q.sub(7).collapse()
Q_mu_n_12 = Q.sub(8).collapse()

# function space for the external force
Q_f = VectorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'], dim=2)

Q_X = VectorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'], dim=2)

# Define functions
#the Jacobian
J_psi = TrialFunction(Q)
psi = Function(Q)
nu_v_bar, nu_w_bar, nu_phi, nu_v_n, nu_w_n, nu_u_n_12, nu_nu_n_12, nu_psi_n_12,  nu_mu_n_12 = TestFunctions( Q )

#fields at the preceeding steps
v_n_1 = Function(Q_v_n)
v_n_2 = Function(Q_v_n)
w_n_1 = Function(Q_w_n)
sigma_n_12 = Function( Q_phi )
sigma_n_32 = Function( Q_phi )
sigma_n_12_output = Function( Q_phi )
u_n_32 = Function( Q_u_n_12 )

f = Function(Q_f)
X_ref = Function(Q_X)

#these functions are used to print the solution to file
v_bar_output= Function(Q_v_bar)
w_bar_output = Function(Q_w_bar)
phi_output = Function(Q_phi)
v_n_output = Function(Q_v_n)
w_n_output = Function(Q_w_n)
u_n_12_output = Function(Q_u_n_12)
nu_n_12_output = Function(Q_nu_n_12)
psi_n_12_output = Function(Q_psi_n_12)
mu_n_12_output = Function(Q_mu_n_12)

#vbar_0, ...., z_n_0 are used to store the initial conditions
v_bar_0 = Function( Q_v_bar )
w_bar_0 = Function( Q_w_bar )
phi_0 = Function(Q_phi)
v_n_0 = Function( Q_v_n )
w_n_0 = Function( Q_w_n )
u_n_12_0 = Function( Q_u_n_12)
nu_n_12_0 = Function( Q_nu_n_12 )
psi_n_12_0 = Function( Q_psi_n_12 )
mu_n_12_0 = Function( Q_mu_n_12 )


v_bar, w_bar, phi, v_n, w_n, u_n_12, nu_n_12, psi_n_12, mu_n_12 = split( psi )
V = (v_bar + v_n_1) / 2.0
W = (w_bar + w_n_1) / 2.0

assigner = FunctionAssigner(Q, [Q_v_bar, Q_w_bar, Q_phi, Q_v_n, Q_w_n, Q_u_n_12, Q_nu_n_12, Q_psi_n_12, Q_mu_n_12])
