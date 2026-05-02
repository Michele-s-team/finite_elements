'''
The fields are

- v_n[alpha] = {v^n_alpha}_notes
- v_n_1[alpha] = {v^{n-1}_alpha}_notes
- v_n_2[alpha] = {v^{n-2}_alpha}_notes

- v_[alpha] = {\overline{v}_alpha}_notes

- sigma_n_12 = \sigma^{n-1/2}_notes
- sigma_n_32 = \sigma^{n-3/2}_notes

- phi= \phi_notes

'''

from fenics import *

import mesh.load as lmsh


D_v_ = VectorElement('DG', triangle, 2)
D_phi = VectorElement('DG', triangle, 1)
D_v_n = VectorElement('DG', triangle, 2)

element = MixedElement([D_v_, D_phi, D_v_n])

Q = FunctionSpace(lmsh.mesh, element)

Q_v_ = Q.sub(0).collapse()
Q_phi = Q.sub(1).collapse()
Q_v_n = Q.sub(2).collapse()

Q_f = VectorFunctionSpace(lmsh.mesh, 'DG', 2)
Q_tau = VectorFunctionSpace(lmsh.mesh, 'DG', 2)


# Define functions for solutions at previous and current time steps
psi = Function(Q)
v_, phi, v_n = split(psi)

v_n_1 = Function(Q_v_n)
v_n_2 = Function(Q_v_n)

sigma_n_12 = Function(Q_phi)
sigma_n_32 = Function(Q_phi)

f = Function(Q_f)
tau = Function(Q_f)

# velocity profiles for the BCs
v_l = Function(Q_v_)
v_tb_circle = Function(Q_v_)

# Define test functions
nu_v_, nu_phi, nu_v_n = TestFunctions(Q)

J_psi = TrialFunction(Q)

V = 0.5 * (v_n_1 + v_)
