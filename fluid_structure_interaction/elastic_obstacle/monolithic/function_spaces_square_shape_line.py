from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the variables for the problem are

    - 'u_n': u^n in notes
    - 'u_dot_n', 'u_dot_n_1': \dot{u}^n, \dot{u}^{n-1} in notes
    - v_n, v_n_1 : \textrm{v}^n_notes, \textrm{v}^{n-1}_notes
    - 'sigma_n' = \varsigma^n_notes

    - 'u_0' = u_0_{Decomposition of deformation field}

all fields are defined from a mixed function space
'''

#1 define elements 

#1.1 fluid 
D_v_n = VectorElement('DG', triangle, 2)
D_sigma_n = FiniteElement('DG', triangle, 1)

#1.2 elastic body and mesh
D_u = VectorElement('DG', triangle, rpam.parameters['u_function_space_degree'])
D_u_dot = VectorElement('DG', triangle, rpam.parameters['u_dot_function_space_degree'])

element = MixedElement([D_v_n, D_sigma_n, D_u, D_u_dot])



#2 define function spaces

#2.1 global function space
Q = FunctionSpace(lmsh.mesh[0], element)

#2.2 collapsed function spaces
Q_v_n = Q.sub(0).collapse()
Q_sigma_n = Q.sub(1).collapse()

Q_u_n = Q.sub(2).collapse()
Q_u_dot_n = Q.sub(3).collapse()

Q_rho_el = FunctionSpace(lmsh.mesh[0], 'DG', 1)
Q_det_F = FunctionSpace(lmsh.mesh[0], 'DG', 1)



#3 define fields

# 3.1 psi contains all fields
psi = Function(Q)
v_n, sigma_n, u_n, u_dot_n = split(psi)


# 3.2 auxiliary fields
v_n_1 = Function(Q_v_n)

u_n_1 = Function(Q_u_n)
u_dot_n_1 = Function(Q_u_dot_n)

rho_el = Function(Q_rho_el)

sigma_r = Function(Q_sigma_n)

u_0 = Function(Q_u_n)


# y is the identity function that, given the coordinates y_i in the reference configuration, returns y_i
y = Function(Q_u_n)

# 3.2.1 fields to store initial condition read from file

v_input = Function(Q_v_n)
sigma_input = Function(Q_sigma_n)
u_input = Function(Q_u_n)
u_dot_input = Function(Q_u_dot_n)



# velocity profiles for the BCs
f = Function(Q_v_n)
v_l = Function(Q_v_n)
v_tb = Function(Q_v_n)



# 3.3 test functions
nu_v_n, nu_sigma_n, nu_u_n, nu_u_dot_n = TestFunctions(Q)

# 3.4 jacobian
J_psi = TrialFunction(Q)

# 3.5 function assigner

assigner = FunctionAssigner(Q, [Q_v_n, Q_sigma_n, Q_u_n, Q_u_dot_n])


