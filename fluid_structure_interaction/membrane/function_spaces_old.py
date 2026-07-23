from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam


from periodic_bc import PeriodicBoundary

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
    
    - 'var_tensor_sigma_fl[alpha, beta]' = {\varsigma_FL_{alpha beta}}_{'channel flow with membrane'}
'''

################## Define function spaces ##################

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

pbc = PeriodicBoundary()
Q_mem = FunctionSpace(lmsh.sub_meshes[1], element_mem,  constrained_domain=pbc)

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

# function space to store the time derivative of U_n_12
Q_U_dot_n_12 = VectorFunctionSpace(lmsh.sub_meshes[1], 'P', rpam.parameters['function_space_degree_mem'], dim=2)

# tensor function space to project the stress tensor of the fluid mesh on the membrane mesh
Q_var_tensor_sigma_fl_on_mem = TensorFunctionSpace(lmsh.sub_meshes[1], 'P', rpam.parameters['function_space_degree_mem'], shape=(2,2))



# 2) for the fictitious elastic body: 
Q_u = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 1)
Q_u_dot = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 1)



# 3) for the fluid:   
Q_v_fl = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 2)
Q_v_fl_bar = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 2)
Q_phi_fl = FunctionSpace(lmsh.sub_meshes[0], 'P', 1)
  
Q_var_tensor_sigma_fl = TensorFunctionSpace(lmsh.sub_meshes[0], 'P', rpam.parameters['function_space_degree_mem'], shape=(2,2))



################## define fields ##################

# 1) for the membrane:
#Jacobian
J_psi_mem = TrialFunction(Q_mem)
psi_mem = Function(Q_mem)

# test functions
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

# function to store the fluid stress tensor projected on the membrane
var_tensor_sigma_fl_on_mem = Function(Q_var_tensor_sigma_fl_on_mem)

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


U_dot_n_12 = Function( Q_U_dot_n_12)


# fields for the boundary conditions
v_bar_l = Function( Q_v_bar )
v_bar_r = Function( Q_v_bar )


# functions derived from psi_mem
v_bar, w_bar, phi, v_n, w_n, U_n_12, nu_n_12, psi_n_12, mu_n_12 = split( psi_mem )

V = (v_bar + v_n_1) / 2.0
W = (w_bar + w_n_1) / 2.0

assigner_mem = FunctionAssigner(Q_mem, [Q_v_bar, Q_w_bar, Q_phi, Q_v_n, Q_w_n, Q_U_n_12, Q_nu_n_12, Q_psi_n_12, Q_mu_n_12])


# 2) for the fictitious elastic body:
u_n = Function(Q_u)
u_n_1 = Function(Q_u)
u_n_2 = Function(Q_u)

u_dot_n = Function(Q_u_dot)
u_dot_n_1 = Function(Q_u_dot)
u_dot_n_2 = Function(Q_u_dot)

# function space to store the projection of the membrane deformation field and of its derivative on the mesh
U_n_12_on_mesh = Function(Q_u)
U_dot_n_12_on_mesh = Function(Q_u_dot)


# jacobians
J_u = TrialFunction(Q_u)
J_u_dot = TrialFunction(Q_u_dot)

# test functions
nu_u = TestFunction(Q_u)
nu_u_dot = TestFunction(Q_u_dot)


# 3) for the fluid
v_fl_n = Function(Q_v_fl)
v_fl_n_1 = Function(Q_v_fl)
v_fl_n_2 = Function(Q_v_fl)
v_fl_bar = Function(Q_v_fl_bar)
sigma_fl_n_12 = Function(Q_phi_fl)
sigma_fl_n_32 = Function(Q_phi_fl)
phi_fl = Function(Q_phi_fl)

# stress tensor of the fluid 
var_tensor_sigma_fl= Function(Q_var_tensor_sigma_fl)

# fields to store the BCs
v_fl_bar_b = Function(Q_v_fl_bar)



# jacobians
J_v_fl_bar = TrialFunction(Q_v_fl_bar)
J_v_fl_n = TrialFunction(Q_v_fl)
J_phi_fl = TrialFunction(Q_phi_fl)

# test functions
nu_v_fl_n = TestFunction(Q_v_fl)
nu_v_fl_bar = TestFunction(Q_v_fl_bar)
nu_phi_fl = TestFunction(Q_phi_fl)

V_fl =  (v_fl_n_1 + v_fl_bar) / 2.0
