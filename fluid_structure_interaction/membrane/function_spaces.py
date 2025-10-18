from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam

'''
the variables for the problem are
1) for the membrane: 
    - v_bar = \overline{v}_{'channel flow with membrane'}
    - w_bar = \overline{w}_{'channel flow with membrane'}
    - phi = phi_{'channel flow with membrane'}
    - v_n = {v^n}_{'channel flow with membrane'}
    - w_n = {w^n}_{'channel flow with membrane'}
    - U_n_12 = {U^{n-1/2}}_{'channel flow with membrane'} or X_n_12 = {X^{n-1/2}}_{'channel flow with membrane'}
    - nu_n_12 = {nu^{n-1/2}}_{'channel flow with membrane'}
    - psi_n_12 = {psi^{n-1/2}}_{'channel flow with membrane'}
    - mu_n_12 = {mu^{n-1/2}}_{'channel flow with membrane'}
    
    - sigma_n_12 = {sigma^{n-1/2}}_{'channel flow with membrane'}

2) for the fictitious elastic body: 
    - u_n = {u^n}_{'channel flow with membrane'}
    - u_dot_n = {\dot{u}^n}_{'channel flow with membrane'}

3) for the fluid:     
    - 'v_fl_n' = {\textrm{v_FL}^n}_{'channel flow with membrane'}
    - 'v_fl_bar' = {\overline{v_FL}}_{'channel flow with membrane'}
    - 'sigma_fl_n_12' = {\varsigma_FL^{n-1/2}}_{'channel flow with membrane'}
    - 'phi_fl' = {phi_FL}_{'channel flow with membrane'}
'''

# Define function spaces
# 1) for the membrane: 
# mixed funtion space
P_v_bar = VectorElement( 'P', interval, 2 )
P_w_bar = FiniteElement( 'P', interval, 1 )
P_phi = FiniteElement('P', interval, 1)
P_v_n = VectorElement( 'P', interval, 2 )
P_w_n = FiniteElement( 'P', interval, 1 )
P_U_n_12 = VectorElement('P', interval, rpam.parameters['function_space_degree_mem'], dim=2)
P_nu_n_12 = FiniteElement('P', interval, rpam.parameters['function_space_degree_mem'])
P_psi_n_12 = FiniteElement('P', interval, rpam.parameters['function_space_degree_mem'])
P_mu_n_12 = FiniteElement( 'P', interval, rpam.parameters['function_space_degree_mem'] )

element_mem = MixedElement( [P_v_bar, P_w_bar, P_phi, P_v_n, P_w_n, P_U_n_12, P_nu_n_12, P_psi_n_12, P_mu_n_12] )
Q_mem = FunctionSpace(lmsh.sub_meshes[1], element_mem)

# collapsed function spaces
Q_v_bar = Q_mem.sub(0).collapse()
Q_w_bar = Q_mem.sub(1).collapse()
Q_phi = Q_mem.sub(2).collapse()
Q_v_n = Q_mem.sub(3).collapse()
Q_w_n = Q_mem.sub(4).collapse()
Q_U_n_12 = Q_mem.sub(5).collapse()
Q_nu_n_12 = Q_mem.sub(6).collapse()
Q_psi_n_12 = Q_mem.sub(7).collapse()
Q_mu_n_12 = Q_mem.sub(8).collapse()

# function space for the X field
Q_X = VectorFunctionSpace(lmsh.sub_meshes[1], 'P', rpam.parameters['function_space_degree_mem'], dim=2)



# 2) for the fictitious elastic body: 
P_u = VectorElement( 'P', triangle, 1 )
P_u_dot = VectorElement( 'P', triangle, 1 )

element_el = MixedElement( [P_u, P_u_dot] )
Q_el = FunctionSpace(lmsh.sub_meshes[0], element_el)


# 3) for the fluid:   
Q_v_fl = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 2)
Q_v_fl_bar = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 2)
Q_phi_fl = FunctionSpace(lmsh.sub_meshes[0], 'P', 1)
  


# define fields 
# 1) for the membrane:
#Jacobian
J_psi_mem = TrialFunction(Q_mem)
psi_mem = Function(Q_mem)
nu_v_bar, nu_w_bar, nu_phi, nu_v_n, nu_w_n, nu_U_n_12, nu_nu_n_12, nu_psi_n_12,  nu_mu_n_12 = TestFunctions( Q_mem )

#fields at the preceeding steps
v_n_1 = Function(Q_v_n)
v_n_2 = Function(Q_v_n)
w_n_1 = Function(Q_w_n)

sigma_n_12 = Function( Q_phi )
sigma_n_32 = Function( Q_phi )
sigma_n_12_output = Function( Q_phi )

U_n_32 = Function( Q_U_n_12 )

#reference configuration
X_ref = Function(Q_X)

#these functions are used to print the solution to file
v_bar_output = Function(Q_v_bar)
w_bar_output = Function(Q_w_bar)
phi_output = Function(Q_phi)
v_n_output = Function(Q_v_n)
w_n_output = Function(Q_w_n)
U_n_12_output = Function(Q_U_n_12)
nu_n_12_output = Function(Q_nu_n_12)
psi_n_12_output = Function(Q_psi_n_12)
mu_n_12_output = Function(Q_mu_n_12)

#fields to store the initial conditions
v_bar_0 = Function( Q_v_bar )
w_bar_0 = Function( Q_w_bar )
phi_0 = Function( Q_phi )
v_n_0 = Function( Q_v_n )
w_n_0 = Function( Q_w_n )
U_n_12_0 = Function( Q_U_n_12)
nu_n_12_0 = Function( Q_nu_n_12 )
psi_n_12_0 = Function( Q_psi_n_12 )
mu_n_12_0 = Function( Q_mu_n_12 )


# Define functions for solutions at previous and current time steps
v_n = Function(Q_v)
v_n_1 = Function(Q_v)
v_n_2 = Function(Q_v)
v_ = Function(Q_v_)
# sigma^{n-1/2}
sigma_n_12 = Function(Q_phi)
# sigma^{n-3/2}
sigma_n_32 = Function(Q_phi)
phi = Function(Q_phi)
u_n = Function(Q_u)
u_dot_n = Function(Q_u_dot)
u_n_1 = Function(Q_u)
u_dot_n_1 = Function(Q_u_dot)
u_n_2 = Function(Q_u)
u_dot_n_2 = Function(Q_u_dot)

u_ellipse = Function(Q_u)
u_square = Function(Q_u)
u_dot_ellipse = Function(Q_u_dot)
u_dot_square = Function(Q_u_dot)

# y_ellipse = {y^s}_notes
ys_ellipse = Function(Q_y)
# dyds_ellipse = {dy^s/ds}_notes
dyds_ellipse = Function(Q_dyds)

# Define test functions
nu_v_n = TestFunction(Q_v)
nu_v_ = TestFunction(Q_v_)
nu_phi = TestFunction(Q_phi)
nu_u = TestFunction(Q_u)
nu_u_dot = TestFunction(Q_u_dot)


# Jacobians
J_v_ = TrialFunction(Q_v_)
J_v_n = TrialFunction(Q_v)
J_phi = TrialFunction(Q_phi)
J_u = TrialFunction(Q_u)
J_u_dot = TrialFunction(Q_u_dot)

V = 0.5 * (v_n_1 + v_)
