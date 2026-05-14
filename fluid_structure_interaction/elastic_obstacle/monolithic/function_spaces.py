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
Q = FunctionSpace(lmsh.mesh, element)

#2.2 collapsed function spaces
Q_v_n = Q.sub(0).collapse()
Q_sigma_n = Q.sub(1).collapse()

Q_u_n = Q.sub(2).collapse()
Q_u_dot_n = Q.sub(3).collapse()

Q_rho_el = FunctionSpace(lmsh.mesh, 'DG', 1)
Q_det_F = FunctionSpace(lmsh.mesh, 'DG', 1)



#3 define fields

# 3.1 psi contains all fields
psi = Function(Q)
v_n, sigma_n, u_n, u_dot_n = split(psi)


# 3.2 auxiliary fields
v_n_1 = Function(Q_v_n)

u_n_1 = Function(Q_u_n)
u_n_2 = Function(Q_u_n)

u_dot_n_1 = Function(Q_u_dot_n)
u_dot_n_2 = Function(Q_u_dot_n)

rho_el = Function(Q_rho_el)

sigma_r = Function(Q_sigma_n)

# 3.1.1 fields for reading from file
sigma_input = Function(Q_sigma_n)
v_input = Function(Q_v_n)


# velocity profiles for the BCs
f = Function(Q_v_n)
v_l = Function(Q_v_n)
v_tb = Function(Q_v_n)



# 3.3 test functions
nu_v_n, nu_sigma_n, nu_u_n, nu_u_dot_n = TestFunctions(Q)

# 3.4 jacobian
J_psi = TrialFunction(Q)


