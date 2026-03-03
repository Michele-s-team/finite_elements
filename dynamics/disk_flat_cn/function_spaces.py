'''
The fields are

- v_n[alpha] = {v^n_alpha}_notes
- v_n_1[alpha] = {v^{n-1}_alpha}_notes
- v_n_2[alpha] = {v^{n-2}_alpha}_notes

- v_[alpha] = {\overline{v}_alpha}_notes

- sigma_n_12 = \sigma^{n-1/2}_notes
- sigma_n_32 = \sigma^{n-3/2}_notes

- phi= \phi_notes

- omega[alpha] = {\omega_\alpha}_notes
'''

from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam

# Define function spaces
# the '2' in ''P', 2)' is the order of the polynomials used to describe these spaces: if they are low, then derivatives high enough of the functions projected on thee spaces will be set to zero !
Q_v = VectorFunctionSpace(lmsh.mesh, 'P', 2, dim=2)
Q_sigma = FunctionSpace(lmsh.mesh, 'P', rpam.parameters['sigma_function_space_degree'])
Q_omega = VectorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['sigma_function_space_degree'])

Q_f = VectorFunctionSpace(lmsh.mesh, 'P', 2)
Q_tau = VectorFunctionSpace(lmsh.mesh, 'P', 2)


# Define functions for solutions at previous and current time steps
v_n = Function(Q_v)
v_n_1 = Function(Q_v)
v_n_2 = Function(Q_v)v_ = Function(Q_v)
sigma_n_12 = Function(Q_sigma)
sigma_n_32 = Function(Q_sigma)
phi = Function(Q_sigma)
omega = Function(Q_omega)

f = Function(Q_f)
tau = Function(Q_f)

# Define test functions
nu_v_ = TestFunction(Q_v)
nu_v_n = TestFunction(Q_v)
nu_phi = TestFunction(Q_sigma)
nu_omega = TestFunction(Q_omega)

J_v_ = TrialFunction(Q_v)
J_v_n = TrialFunction(Q_v)
J_phi = TrialFunction(Q_sigma)

V = 0.5 * (v_n_1 + v_)
