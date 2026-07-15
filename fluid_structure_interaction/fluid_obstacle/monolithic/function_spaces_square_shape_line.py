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
    - 'u_n', 'u_n_1: u^n, u^{n-1} in notes
    - 'u_dot_n', 'u_dot_n_1': \dot{u}^n, \dot{u}^{n-1} in notes
    - 'c_n', 'c_n_1' = \textrm{c}^n_notes, \textrm{c}^{n-1}_notes

    - `v_lrb` = {g^n}_notes
    - `f_shape` = {\textrm{f}^circle}_notes
    - `f_square` = {\textrm{f}^square}_notes
    - `t_t` = {\textrm{t}^n}_notes (traction on ds_t)
    - `sigma_square_t` = {sigma_{square T}}_notes
    - `dyds` = {d y_s / ds}_notes for \partial \Omega_O

    - 'mu_n': the curvature of the shape curve x_s in the current configuration
    - 'grad_u_n': grad_u_n[i, j] = \partial u_n_i / \partial y_j
    - f = f_{Curvature} tangent vector to the curve y_s in the reference configuration, extended to the whole domain
    - e = e_{Curvature}, tangent vector to the curve x_s in the current configuration, extended to the whole domain
    - nu = nu_{Curvature} unit normal to y_s pointing outwards \Omega_circle^y, extended to the whole domain
    - n = n_{Curvature} unit normal to x_s pointing inwards \Omega_circle^y, extended to the whole domain


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

# 1.4 surface tension
D_mu_n = FiniteElement('DG', triangle, rpam.parameters['u_function_space_degree'])
D_grad_u_n = TensorElement('DG', triangle, rpam.parameters['u_function_space_degree'])



element = MixedElement([D_v_n, D_sigma_n, D_u_n, D_u_dot_n, D_c_n, D_mu_n, D_grad_u_n])



#2 define function spaces

#2.1 global function space
Q = FunctionSpace(lmsh.mesh[0], element)

#2.2 collapsed function spaces
Q_v_n = Q.sub(0).collapse()
Q_sigma_n = Q.sub(1).collapse()

Q_u_n = Q.sub(2).collapse()
Q_u_dot_n = Q.sub(3).collapse()

Q_c_n = Q.sub(4).collapse()

Q_mu_n = Q.sub(5).collapse()
Q_grad_u_n = Q.sub(6).collapse()



# 2.3 auxiliary function spaces
V = VectorFunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['u_function_space_degree'])
Q_f = VectorFunctionSpace(lmsh.mesh[0], 'DG', rpam.parameters['f_function_space_degree'])


#3 fields

# 3.1 psi contains all fields
psi = Function(Q)
v_n, sigma_n, u_n, u_dot_n, c_n, mu_n, grad_u_n = split(psi)



# 3.2 auxiliary fields
v_n_1 = Function(Q_v_n)
u_n_1 = Function(Q_u_n)
c_n_1 = Function(Q_c_n)

f_shape = Function(Q_f)
f_square = Function(Q_f)

dyds = Function(Q_u_n)

# 3.2.1 fields for BCs
v_lrb = Function(Q_v_n) 
t_t = Function(Q_f)
sigma_square_t = Function(Q_sigma_n)

#3.2.3 fields for the curvature computation

f = Function(V)
nu = Function(V)
b = Function(Q_grad_u_n)


# 3.3 test functions
nu_v_n, nu_sigma_n, nu_u_n, nu_u_dot_n, nu_c_n, nu_mu_n, nu_grad_u_n = TestFunctions(Q)


# 3.4 jacobian
J_psi = TrialFunction(Q)

# 3.5 function assigner

assigner = FunctionAssigner(Q, [Q_v_n, Q_sigma_n, Q_u_n, Q_u_dot_n, Q_c_n, Q_mu_n, Q_grad_u_n])