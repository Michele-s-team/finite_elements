from fenics import *

import mesh.load as lmsh

'''
the variables for the problem are
1) for the membrane: 
    - v_n = {v^n}_{'channel flow with membrane'}
    - w_n = {w^n}_{'channel flow with membrane'}
    - v_bar = \overline{v}_{'channel flow with membrane'}
    - w_bar = \overline{w}_{'channel flow with membrane'}
    - phi = phi_{'channel flow with membrane'}
    - sigma_n_12 = {sigma^{n-1/2}}_{'channel flow with membrane'}
    - X^{n-1/2} = {X^{n-1/2}}_{'channel flow with membrane'}
    - nu_n_12 = {nu^{n-1/2}}_{'channel flow with membrane'}
    - psi_n_12 = {psi^{n-1/2}}_{'channel flow with membrane'}
    - mu_n_12 = {mu^{n-1/2}}_{'channel flow with membrane'}
    
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
Q_v = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q_v_ = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q_phi = FunctionSpace(lmsh.mesh, 'P', 1)
Q_u = VectorFunctionSpace(lmsh.mesh, 'P', 1)
Q_u_dot = VectorFunctionSpace(lmsh.mesh, 'P', 1)

# function space for the vector dy(s)/ds which represents the tangent to the ellipse curve
Q_y = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q_dyds = VectorFunctionSpace(lmsh.mesh, 'P', 2)



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
