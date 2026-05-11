from fenics import *
import importlib

import mesh.load as lmsh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the variables for the problem are

    - 'u_n', 'u_n_1', 'u_n_2': u^n, u^{n-1}, u^{n-2} in notes
    - 'u_dot_n', 'u_dot_n_1', 'u_dot_n_2' : u^n, u^{n-1}, u^{n-2} in notes
    - 'v^n' = \textrm{v}^n_notes
    - 'sigma_n_1', 'sigma_n_2' = \varsigma^{n-1}_notes,  \varsigma^{n-2}_notes

all fields are defined from a mixed function space
'''

#1 define elements 

#1.1 fluid 
D_v_n = VectorElement('DG', triangle, 2)
D_sigma_n = FiniteElement('DG', triangle, 1)

#1.2 elastic body and mesh
D_u = VectorElement('DG', triangle, 1)
D_u_dot = VectorElement('DG', triangle, 1)

element = MixedElement([D_v_n, D_sigma_n, D_u, D_u_dot])



#2 define function spaces

#2.1 global function space
Q = FunctionSpace(lmsh.mesh, element)

#2.2 collapsed function spaces
Q_v_n = Q.sub(0).collapse()
Q_sigma_n = Q.sub(1).collapse()

Q_u = Q.sub(2).collapse()
Q_u_dot = Q.sub(3).collapse()

Q_rho_el = FunctionSpace(lmsh.mesh, 'DG', 1)

Q_dyds = VectorFunctionSpace(lmsh.mesh, 'DG', 2)


#3 define fields

# 3.1 psi contains all fields
psi = Function(Q)
v_n, sigma_n, u_n, u_dot_n = split(psi)


# 3.2 auxiliary fields
v_n_1 = Function(Q_v_n)

u_n_1 = Function(Q_u)
u_n_2 = Function(Q_u)

u_dot_n_1 = Function(Q_u_dot)
u_dot_n_2 = Function(Q_u_dot)

rho_el = Function(Q_rho_el)

# dyds_ellipse = {dy^s/ds}_notes
dyds_ellipse = Function(Q_dyds)

# velocity profiles for the BCs
f = Function(Q_v_n)
v_l = Function(Q_v_n)
v_tb = Function(Q_v_n)



# 3.3 test functions
nu_v_n, nu_sigma_n, nu_u_n, nu_u_dot_n = TestFunctions(Q)

# 3.4 jacobian
J_psi = TrialFunction(Q)


