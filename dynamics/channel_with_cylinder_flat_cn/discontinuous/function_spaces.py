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


Q_v = VectorFunctionSpace(lmsh.mesh, 'DG', 2)
Q_v_ = VectorFunctionSpace(lmsh.mesh, 'DG', 2)

Q_sigma = FunctionSpace(lmsh.mesh, 'DG', 1)

Q_f = VectorFunctionSpace(lmsh.mesh, 'DG', 2)
Q_tau = VectorFunctionSpace(lmsh.mesh, 'DG', 2)


# Define functions for solutions at previous and current time steps
v_n = Function(Q_v)
v_n_1 = Function(Q_v)
v_n_2 = Function(Q_v)

v_ = Function(Q_v_)

sigma_n_12 = Function(Q_sigma)
sigma_n_32 = Function(Q_sigma)

phi = Function(Q_sigma)

f = Function(Q_f)
tau = Function(Q_f)

# velocity profiles for the BCs
v_l = Function(Q_v_)
v_tb_circle = Function(Q_v_)

# Define test functions
nu_v_ = TestFunction(Q_v_)
nu_v_n = TestFunction(Q_v)
nu_phi = TestFunctions(Q_sigma)


J_v_ = TrialFunction(Q_v_)
J_v_n = TrialFunction(Q_v)
J_phi = TrialFunction(Q_sigma)

V = 0.5 * (v_n_1 + v_)
