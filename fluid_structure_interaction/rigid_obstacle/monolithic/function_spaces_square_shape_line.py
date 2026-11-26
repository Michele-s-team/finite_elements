from fenics import *
import importlib

import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the variables for the problem are

# 1. body

    - 'theta_n' = \theta_n in 'Channel flow with moving obstacle'
    - 'omega_n' = \omega_n in 'Channel flow with moving obstacle'

# 2. fluid

    - v_n, v_n_1 : \textrm{v}^n, \textrm{v}^{n-1} in 'Channel flow with moving obstacle'
    - 'sigma_n' = \varsigma^n in 'Channel flow with moving obstacle'

# 3. domain (mesh)

    - 'u_n': u^n in 'Channel flow with moving obstacle'
    - 'u_dot_n', 'u_dot_n_1': \dot{u}^n, \dot{u}^{n-1} in 'Channel flow with moving obstacle'

all fields are defined from a mixed function space
'''


#1. define elements 

# 1.1 body

R_theta_n = FiniteElement('Real', msh.element_geometry(lmsh.mesh[0]), 0)
R_omega_n = FiniteElement('Real', msh.element_geometry(lmsh.mesh[0]), 0)


#1.2 fluid 

D_v_n = VectorElement('DG', msh.element_geometry(lmsh.mesh[0]), 2)
D_sigma_n = FiniteElement('DG', msh.element_geometry(lmsh.mesh[0]), 1)


#1.3 domain

D_u_n = VectorElement('DG',  msh.element_geometry(lmsh.mesh[0]), rpam.parameters['u_function_space_degree'])
D_u_dot_n = VectorElement('DG',  msh.element_geometry(lmsh.mesh[0]), rpam.parameters['u_dot_function_space_degree'])

element = MixedElement([R_theta_n, R_omega_n, D_v_n, D_sigma_n, D_u_n, D_u_dot_n])


#2 define function spaces

#2.1 global function space
Q = FunctionSpace(lmsh.mesh[0], element)


#2.2 collapsed function spaces

# 2.2.1 body

Q_theta_n = Q.sub(0).collapse()
Q_omega_n = Q.sub(1).collapse()

# 2.2.2 fluid

Q_v_n = Q.sub(2).collapse()
Q_sigma_n = Q.sub(3).collapse()

# 2.2.3 domain

Q_u_n = Q.sub(4).collapse()
Q_u_dot_n = Q.sub(5).collapse()



#3 define fields

# 3.1 psi contains all fields
psi = Function(Q)
theta_n, omega_n, v_n, sigma_n, u_n, u_dot_n = split(psi)




# 3.2 auxiliary fields

# 3.2.1 body

theta_n_1 = Function(Q_theta_n)
omega_n_1 = Function(Q_omega_n)

theta_0 = Function(Q_theta_n)
omega_0 = Function(Q_omega_n)


# 3.2.2 fluid

v_n_1 = Function(Q_v_n)

v_0 = Function(Q_v_n)
sigma_0 = Function (Q_sigma_n)

# velocity profiles for the BCs

v_l = Function(Q_v_n)
v_tb = Function(Q_v_n)


# 3.2.3 domain

u_n_1 = Function(Q_u_n)
u_dot_n_1 = Function(Q_u_dot_n)

u_0 = Function(Q_u_n)
u_dot_0 = Function(Q_u_dot_n)


# y is the identity function that, given the coordinates y_i in the reference configuration, returns y_i
y = Function(Q_u_n)

# y.set_allow_extrapolation(True)

# 3.3 test functions

nu_theta_n, nu_omega_n, nu_v_n, nu_sigma_n, nu_u_n, nu_u_dot_n = TestFunctions(Q)




# 3.4 jacobian
J_psi = TrialFunction(Q)

# 3.5 function assigner

assigner = FunctionAssigner(Q, [Q_theta_n, Q_omega_n, Q_v_n, Q_sigma_n, Q_u_n, Q_u_dot_n])