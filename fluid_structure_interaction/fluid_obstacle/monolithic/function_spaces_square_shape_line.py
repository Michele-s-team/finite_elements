from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the variables for the problem are
    - `v_n`, `v_n_1` : \textrm{v}^n_notes, \textrm{v}^{n-1}_notes
    - 'sigma_n' = \varsigma^n_notes
    - 'u_n': u^n in notes
    - 'u_dot_n', 'u_dot_n_1': \dot{u}^n, \dot{u}^{n-1} in notes
    - 'c_n' = \textrm{c}^n_notes

    - `v_lrb` = {g^n}_notes
    - `f_shape` = {\textrm{f}^circle}_notes
    - `f_square` = {\textrm{f}^square}_notes
    - `t_t` = {\textrm{t}^n}_notes (traction on ds_t)
'''

#1 define elements 

#1.1 fluid 
D_v_n = VectorElement('DG', triangle, 2)
D_sigma_n = FiniteElement('DG', triangle, 1)

#1.2 mesh
D_u_n = VectorElement('DG', triangle, rpam.parameters['u_function_space_degree'])
D_u_dot_n = VectorElement('DG', triangle, rpam.parameters['u_dot_function_space_degree'])

#1.3 concentration
D_c_n = FiniteElement('DG', triangle, rpam.parameters['c_function_space_degree'])


element = MixedElement([D_v_n, D_sigma_n, D_u_n, D_u_dot_n, D_c_n])



#2 define function spaces

#2.1 global function space
Q = FunctionSpace(lmsh.mesh[0], element)

#2.2 collapsed function spaces
Q_v_n = Q.sub(0).collapse()
Q_sigma_n = Q.sub(1).collapse()

Q_u_n = Q.sub(2).collapse()
Q_u_dot_n = Q.sub(3).collapse()

Q_c_n = Q.sub(4).collapse()

# 2.3 auxiliary function spaces
Q_f = VectorFunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['f_function_space_degree'])



#3 define fields

# 3.1 psi contains all fields
psi = Function(Q)
v_n, sigma_n, u_n, u_dot_n, c_n = split(psi)

# sign


# 3.2 auxiliary fields
v_n_1 = Function(Q_v_n)
u_n_1 = Function(Q_u_n)

f_shape = Function(Q_f)
f_square = Function(Q_f)
t_t = Function(Q_f)

# 3.2.1 fields for BCs
v_lrb = Function(Q_v_n) 


# 3.3 test functions
nu_v_n, nu_sigma_n, nu_u_n, nu_u_dot_n, nu_c_n = TestFunctions(Q)


# 3.4 jacobian
J_psi = TrialFunction(Q)

# 3.5 function assigner

assigner = FunctionAssigner(Q, [Q_v_n, Q_sigma_n, Q_u_n, Q_u_dot_n, Q_c_n])




